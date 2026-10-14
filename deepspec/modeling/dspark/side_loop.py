"""Layer-1 side recurrence beside an unchanged five-layer DFlash stack.

``fusion`` has no intermediate vocabulary projection. ``grouped`` greedily
decodes provisional groups, feeds their embeddings back with the repository's
next-token shift, and fuses pre-feedback states at the end. Final proposals are
resampled from the fused head; provisional tokens are NOT the final AR prefix.
"""
from __future__ import annotations

from contextlib import contextmanager
import math

import torch
from torch import nn

from deepspec.modeling.dspark.context_cache import ContextKVCache


SIDE_CONFIG_DEFAULTS = {
    "side_loop_mode": "none",
    "side_loop_steps": 5,
    "side_group_size": 3,
    "side_fusion_rank": 128,
    "side_aux_loss_alpha": 0.2,
}


def validate_side_config(config):
    mode = getattr(config, "side_loop_mode", "none")
    if mode not in ("none", "fusion", "grouped"):
        raise ValueError("side_loop_mode must be none|fusion|grouped")
    if mode == "none":
        return
    if int(config.num_hidden_layers) != 5 or int(getattr(config, "num_loops", 1)) != 1:
        raise ValueError("Side recurrence requires five main layers and num_loops=1")
    if int(config.markov_rank) or bool(config.enable_confidence_head) or getattr(config, "enable_exit_gate", False):
        raise ValueError("Side recurrence currently supports plain DFlash only")
    for name in ("side_loop_steps", "side_group_size", "side_fusion_rank"):
        if int(getattr(config, name, SIDE_CONFIG_DEFAULTS[name])) < 1:
            raise ValueError(f"{name} must be positive")
    alpha = float(getattr(config, "side_aux_loss_alpha", 0.2))
    if not math.isfinite(alpha) or alpha < 0:
        raise ValueError("side_aux_loss_alpha must be finite and nonnegative")
    if mode == "grouped" and int(config.block_size) != int(config.side_loop_steps) * int(config.side_group_size):
        raise ValueError("grouped requires block_size = side_loop_steps * side_group_size")


class SideFusion(nn.Module):
    """Low-rank residual MLP; zero output initially preserves the main path."""
    def __init__(self, hidden_size, rank):
        super().__init__()
        self.down = nn.Linear(2 * hidden_size, rank, bias=False)
        self.up = nn.Linear(rank, hidden_size, bias=False)

    def forward(self, main, side):
        return main + self.up(torch.nn.functional.silu(self.down(torch.cat((main, side), dim=-1))))


class PathTimer:
    """CUDA events are resolved by the benchmark, never synchronize a fork."""
    def __init__(self, enabled, device):
        self.enabled = bool(enabled)
        self.device = device
        self.events = {}

    @contextmanager
    def stage(self, name):
        if not self.enabled:
            yield
            return
        if self.device.type != "cuda":
            raise ValueError("Side path event profiling requires CUDA")
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        with torch.profiler.record_function(f"side_loop/{name}"):
            yield
        end.record()
        self.events.setdefault(name, []).append((start, end))

    def resolve(self):
        # Caller must synchronize before resolving.
        return {name + "_ms": sum(a.elapsed_time(b) for a, b in pairs)
                for name, pairs in self.events.items()}


def prepare_paths(model, *, position_ids, noise_embedding, target_hidden_states,
                  attention_mask=None, past_key_values=None, timer=None, **kwargs):
    if past_key_values is not None and not isinstance(past_key_values, ContextKVCache):
        raise ValueError("Side recurrence requires ContextKVCache; block KV must not persist")
    if noise_embedding.shape[1] % model.block_size:
        raise ValueError("Side recurrence requires complete anchor blocks")
    timer = timer or PathTimer(False, noise_embedding.device)
    with timer.stage("prepare"):
        context = model.hidden_norm(model.fc(target_hidden_states))
        rope = model.rotary_emb(noise_embedding, position_ids)
        projected = []
        for index, layer in enumerate(model.layers):
            pair = layer.self_attn.project_context(context, rope)
            if past_key_values is not None:
                pair = past_key_values.update(*pair, index)
            projected.append(pair)
    # Immutable input dictionary; both branches reuse context projections.
    return dict(context=context, rope=rope, projected=projected,
                position_ids=position_ids, attention_mask=attention_mask,
                noise=noise_embedding, layer_kwargs=kwargs, timer=timer)


def apply_layer(model, prepared, index, hidden):
    return model.layers[index](
        hidden_states=hidden, target_hidden_states=prepared["context"],
        position_embeddings=prepared["rope"], position_ids=prepared["position_ids"],
        attention_mask=prepared["attention_mask"], past_key_value=None,
        projected_context_kv=prepared["projected"][index], **prepared["layer_kwargs"],
    )


def shared_first(model, prepared):
    with prepared["timer"].stage("shared_layer1"):
        return apply_layer(model, prepared, 0, prepared["noise"])


def main_tail(model, prepared, first):
    with prepared["timer"].stage("main_tail"):
        hidden = first
        for index in range(1, 5):
            hidden = apply_layer(model, prepared, index, hidden)
        return hidden


def side_branch(model, prepared, first, collect_logits=False):
    with prepared["timer"].stage("side_after_fork"):
        hidden = first
        if model.side_loop_mode == "fusion":
            for _ in range(1, model.side_loop_steps):
                hidden = apply_layer(model, prepared, 0, hidden)
            return hidden, None

        batch, length, width = hidden.shape
        blocks = length // model.block_size
        saved, aux = [], []
        # The shifted block input slot j contains x_j and predicts x_{j+1}.
        # Slot zero always retains the known anchor embedding. No gold labels
        # enter this branch, in either training or evaluation.
        token_inputs = prepared["noise"].reshape(batch, blocks, model.block_size, width)
        for step in range(model.side_loop_steps):
            if step:
                hidden = apply_layer(model, prepared, 0, hidden)
            states = hidden.reshape(batch, blocks, model.block_size, width)
            lo = step * model.side_group_size
            hi = lo + model.side_group_size
            group = states[:, :, lo:hi]
            saved.append(group)  # capture BEFORE any feedback of this group
            with prepared["timer"].stage("intermediate_head"):
                logits = model.compute_logits(model.norm(group))
                ids = logits.detach().argmax(dim=-1)
            if collect_logits:
                aux.append(logits)
            if step + 1 < model.side_loop_steps:
                # x_{lo+1:hi+1} goes into input positions lo+1:hi+1.
                token_inputs = torch.cat((token_inputs[:, :, :lo + 1],
                                          model.embed_tokens(ids),
                                          token_inputs[:, :, hi + 1:]), dim=2)
                known = torch.arange(model.block_size, device=hidden.device) <= hi
                states = torch.where(known[None, None, :, None], token_inputs, states)
                hidden = states.reshape(batch, length, width)
        return torch.cat(saved, dim=2).reshape(batch, length, width), (torch.cat(aux, dim=2) if aux else None)


def run_paths(model, prepared, *, execution="serial", collect_logits=False):
    if execution not in ("serial", "overlap", "main", "side"):
        raise ValueError("execution must be serial|overlap|main|side")
    if execution == "overlap" and (model.training or torch.is_grad_enabled() or prepared["noise"].device.type != "cuda"):
        raise ValueError("overlap requires CUDA eval under no_grad/inference_mode")
    first = shared_first(model, prepared)
    if execution == "main":
        return model.norm(main_tail(model, prepared, first)), None
    if execution == "side":
        side, aux = side_branch(model, prepared, first, collect_logits)
        return model.norm(side), aux
    if execution == "overlap":
        current = torch.cuda.current_stream(first.device)
        if getattr(model, "_side_stream", None) is None:
            model._side_stream = torch.cuda.Stream(device=first.device)
        stream = model._side_stream
        stream.wait_stream(current)
        for tensor in (first, prepared["noise"], prepared["context"], prepared["position_ids"],
                       *prepared["rope"], *(t for pair in prepared["projected"] for t in pair)):
            tensor.record_stream(stream)
        if torch.is_tensor(prepared["attention_mask"]):
            prepared["attention_mask"].record_stream(stream)
        with torch.cuda.stream(stream):
            side, aux = side_branch(model, prepared, first, collect_logits)
            side.record_stream(current)
        main = main_tail(model, prepared, first)
        current.wait_stream(stream)
    else:
        main = main_tail(model, prepared, first)
        side, aux = side_branch(model, prepared, first, collect_logits)
    with prepared["timer"].stage("fusion"):
        output = model.norm(model.side_fusion(main, side))
    return output, aux


def forward_side_paths(model, **kwargs):
    diagnostics = kwargs.pop("diagnostics", False)
    kwargs.pop("diagnostic_mask", None)
    kwargs.pop("use_cache", None)
    kwargs.pop("num_loops", None)
    all_hidden = kwargs.pop("return_all_loop_hidden", False)
    if kwargs.pop("exit_q", None) is not None:
        raise ValueError("Side recurrence cannot use the Ouro exit gate")
    kwargs.pop("min_exit_loop", None)
    timer = PathTimer(bool(getattr(model, "profile_side_paths", False)), kwargs["noise_embedding"].device)
    prepared = prepare_paths(model, timer=timer, **kwargs)
    output, aux = run_paths(model, prepared,
                            execution=getattr(model, "side_execution", "serial"),
                            collect_logits=model.training and model.side_loop_mode == "grouped")
    model._side_aux_logits = aux
    model.last_side_timer = timer
    model.last_exit_depth = 1
    if diagnostics:
        from deepspec.utils.metrics import add_metric
        add_metric("side_loop_steps", model.side_loop_steps, reduction="last", tag="train")
    return (output,) if all_hidden else output

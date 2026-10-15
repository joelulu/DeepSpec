"""Matched-input forward timing on states from real speculative decoding.

Only plain Qwen3 DFlash fusion checkpoints are supported. Graphs use padded,
masked projected-context inputs and fixed buffers. No diagnostic forward writes
to the reference cache. This is per-call latency, not serving throughput.
"""
from __future__ import annotations

import csv
import json
import math
import random
import statistics
import time
from pathlib import Path

import torch

from deepspec.modeling.dspark.context_cache import ContextKVCache
from deepspec.modeling.dspark.side_loop import PathTimer, run_paths


MODES = ("main_only", "side_only", "dual_stream", "fusion_only")
BACKENDS = ("eager", "cudagraph")


class TimingComplete(Exception):
    """Stop reference generation as soon as the matched call budget is met."""


def padding_hidden_report(actual, expected):
    """Validate different-shape BF16 forwards without a single-element veto.

    Padding changes GEMM/SDPA shapes and reduction order. Keep the original
    elementwise test as the first criterion, but allow a sparse, bounded BF16
    tail only when global and per-position relative L2 remain small. This is
    ONLY for padded versus unpadded inputs; graph/eager checks stay unchanged.
    Greedy token equality is checked separately by the caller.
    """
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        raise ValueError("Padded/unpadded hidden shape or dtype differs")
    a, b = actual.float(), expected.float()
    if not bool((torch.isfinite(a).all() & torch.isfinite(b).all()).item()):
        return dict(passed=False, status="nonfinite", dtype=str(actual.dtype))
    delta = a - b
    absolute = delta.abs()
    mismatch = absolute > (0.02 + 0.02 * b.abs())
    count = int(mismatch.sum().item())
    relative_l2 = float((delta.norm() / b.norm().clamp_min(1e-12)).item())
    position_relative_l2 = float((delta.norm(dim=-1) /
                                 b.norm(dim=-1).clamp_min(1e-12)).max().item())
    reference_rms = float(b.square().mean().sqrt().item())
    max_abs = float(absolute.max().item())
    # Eight BF16 epsilon units at the reference RMS scale, with a unit floor
    # appropriate for these final RMS-normalized hidden states.
    max_abs_limit = 8 * torch.finfo(torch.bfloat16).eps * max(1.0, reference_rms)
    fraction = count / actual.numel()
    bounded_tail = (actual.dtype == torch.bfloat16 and fraction <= 0.001
                    and relative_l2 <= 0.01 and position_relative_l2 <= 0.02
                    and max_abs <= max_abs_limit)
    passed = count == 0 or bounded_tail
    return dict(passed=passed, status=("elementwise_close" if count == 0 else
        "bounded_bf16_tail" if bounded_tail else "excessive_error"),
        dtype=str(actual.dtype), elements=actual.numel(), elementwise_mismatches=count,
        mismatch_fraction=fraction, relative_l2=relative_l2,
        max_position_relative_l2=position_relative_l2, reference_rms=reference_rms,
        max_abs=max_abs, limits=dict(mismatch_fraction=0.001, relative_l2=0.01,
            max_position_relative_l2=0.02, max_abs=max_abs_limit))


def bucket_size(length, buckets):
    if length == 0:
        return 0
    for size in buckets:
        if length <= size:
            return size
    raise ValueError(f"Context {length} exceeds largest bucket {buckets[-1]}; "
                     "increase --context-buckets (the prefix is never truncated)")


def distribution(values):
    ordered = sorted(values)
    return dict(mean=statistics.mean(values), median=statistics.median(values),
                p90=ordered[max(0, math.ceil(0.9 * len(ordered)) - 1)])


def summarize_calls(rows):
    summaries = []
    for mode in MODES:
        for backend in BACKENDS:
            selected = [r for r in rows if r["mode"] == mode and r["backend"] == backend]
            if not selected:
                continue
            row = dict(mode=mode, backend=backend, calls=len(selected),
                       context_min=min(r["context_tokens"] for r in selected),
                       context_max=max(r["context_tokens"] for r in selected))
            for metric in ("wall_ms", "gpu_ms", "host_submit_ms", "input_copy_ms"):
                for stat, value in distribution([r[metric] for r in selected]).items():
                    row[f"{metric}_{stat}"] = value
            summaries.append(row)
    for row in summaries:
        base = next(r for r in summaries if r["mode"] == "main_only"
                    and r["backend"] == row["backend"])
        row["extra_wall_ms"] = row["wall_ms_mean"] - base["wall_ms_mean"]
        row["overhead_wall_pct"] = 100 * row["extra_wall_ms"] / base["wall_ms_mean"]
    return summaries


def validate_timing_model(model):
    if len(model.layers) != 5 or model.num_loops != 1:
        raise ValueError("Timing requires five main layers and one main-stack pass")
    if model.side_loop_mode != "fusion" or model.side_fusion is None:
        raise ValueError("Use the trained A/fusion checkpoint for all eight groups")
    if model.markov_head is not None or model.confidence_head is not None:
        raise ValueError("This benchmark supports plain DFlash, without Markov/confidence heads")
    if model.sample_from_anchor or model.block_size != 16:
        raise ValueError("Timing requires block16 = given anchor + 15 predictions")
    if model.training:
        raise ValueError("Call model.eval() before timing")


class StaticDraftInput:
    """Fixed-address, read-only snapshot of an incremental draft input.

    Old KV and newly projected KV are concatenated with separate padding masks.
    This avoids dynamic indexing/scattering inside graph capture. Both backends
    run exactly this layout. Input copies are outside forward timing and reported
    separately; new-feature projection and cache concatenation are inside it.
    """
    def __init__(self, model, old_capacity, new_capacity):
        self.model = model
        self.old_capacity, self.new_capacity = old_capacity, new_capacity
        weight = model.embed_tokens.weight
        device, dtype = weight.device, weight.dtype
        self.device = device
        self.ids = torch.empty((1, model.block_size), device=device, dtype=torch.long)
        self.features = torch.zeros((1, new_capacity,
            len(model.target_layer_ids) * model.config.hidden_size), device=device, dtype=dtype)
        self.positions = torch.zeros((1, new_capacity + model.block_size),
                                     device=device, dtype=torch.long)
        self.old_length = torch.zeros((), device=device, dtype=torch.long)
        self.new_length = torch.zeros((), device=device, dtype=torch.long)
        self.old_indices = torch.arange(old_capacity, device=device)
        self.new_indices = torch.arange(new_capacity, device=device)
        self.block_valid = torch.ones(model.block_size, device=device, dtype=torch.bool)
        self.old_kv = []
        for layer in model.layers:
            attn = layer.self_attn
            shape = (1, attn.num_key_value_heads, old_capacity, attn.head_dim)
            self.old_kv.append((torch.zeros(shape, device=device, dtype=dtype),
                                torch.zeros(shape, device=device, dtype=dtype)))

    def update(self, *, draft_input_ids, target_hidden_states, past_key_values,
               position_ids, start):
        old = past_key_values.get_seq_length()
        new = target_hidden_states.shape[1]
        if old + new != start:
            raise ValueError(f"Context alignment: cached={old}, new={new}, anchor={start}")
        if old > self.old_capacity or new > self.new_capacity:
            raise ValueError("Snapshot capacity exceeded")
        if draft_input_ids.shape != self.ids.shape or position_ids.shape[1] != new + self.model.block_size:
            raise ValueError("Expected one complete draft block and aligned position IDs")
        self.ids.copy_(draft_input_ids)
        self.features.zero_()
        self.features[:, :new].copy_(target_hidden_states)
        self.positions[:, :self.new_capacity].copy_(
            (old + self.new_indices).unsqueeze(0))
        self.positions[:, self.new_capacity:].copy_(position_ids[:, -self.model.block_size:])
        self.old_length.fill_(old)
        self.new_length.fill_(new)
        for index, (key, value) in enumerate(self.old_kv):
            if old:
                if index not in past_key_values.layers:
                    raise ValueError(f"Missing reference context KV at layer {index}")
                old_key, old_value = past_key_values.layers[index]
                if old_key.shape[-2] != old or old_value.shape[-2] != old:
                    raise ValueError("Reference cache layers have unequal lengths")
                key[..., :old, :].copy_(old_key)
                value[..., :old, :].copy_(old_value)

    def forward(self, mode, timer=None):
        if mode not in MODES:
            raise ValueError(f"Unknown timing mode {mode}")
        model = self.model
        timer = timer or PathTimer(False, self.device)
        with timer.stage("prepare"):
            noise = model.embed_tokens(self.ids)
            context = model.hidden_norm(model.fc(self.features))
            rope = model.rotary_emb(noise, self.positions)
            valid = torch.cat((self.old_indices < self.old_length,
                               self.new_indices < self.new_length, self.block_valid))
            mask = valid[None, None, None, :]
            projected = [None] * 5
            needed = (0,) if mode == "side_only" else range(5)
            for index in needed:
                key, value = model.layers[index].self_attn.project_context(context, rope)
                old_key, old_value = self.old_kv[index]
                projected[index] = (torch.cat((old_key, key), dim=-2),
                                    torch.cat((old_value, value), dim=-2))
            prepared = dict(context=context, rope=rope, projected=projected,
                position_ids=self.positions, attention_mask=mask, noise=noise,
                anchors=self.ids[:, :1], layer_kwargs={"is_causal": False}, timer=timer)
        hidden, _ = run_paths(model, prepared, execution=mode)
        with timer.stage("output_head"):
            logits = model.compute_logits(hidden[:, 1:])
        with timer.stage("proposal_heads"):
            tokens, _ = model.sample_draft_tokens(logits,
                first_prev_token_ids=self.ids[:, 0], temperature=0.0,
                hidden_states=hidden[:, 1:])
        return hidden, tokens


class GraphForward:
    def __init__(self, snapshot, mode, warmup=3):
        self.snapshot, self.mode = snapshot, mode
        device = snapshot.device
        stream = torch.cuda.Stream(device=device)
        current = torch.cuda.current_stream(device)
        stream.wait_stream(current)
        with torch.cuda.stream(stream):
            for _ in range(warmup):
                snapshot.forward(mode)
        current.wait_stream(stream)
        torch.cuda.synchronize(device)
        self.graph = torch.cuda.CUDAGraph()
        # run_paths forks from this capture stream and rejoins before returning.
        with torch.cuda.graph(self.graph, stream=stream):
            self.output = snapshot.forward(mode)
        self.graph.replay()
        torch.cuda.synchronize(device)

    def __call__(self):
        self.graph.replay()
        return self.output


def measure_forward(fn, device):
    torch.cuda.synchronize(device)
    begin = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    host_begin = time.perf_counter()
    begin.record()
    output = fn()
    end.record()
    host_submit_ms = (time.perf_counter() - host_begin) * 1000
    end.synchronize()
    wall_ms = (time.perf_counter() - host_begin) * 1000
    return dict(wall_ms=wall_ms, gpu_ms=begin.elapsed_time(end),
                host_submit_ms=host_submit_ms), output


class DraftTimingObserver:
    def __init__(self, model, output, *, warmup_calls=5, measure_calls=50,
                 buckets=(1024, 4096), modes=MODES, backends=BACKENDS,
                 seed=42, profile_trace=False, component_timings=True):
        validate_timing_model(model)
        if warmup_calls < 1 or measure_calls < 1:
            raise ValueError("warmup-calls and measure-calls must be positive")
        if not buckets or tuple(sorted(set(buckets))) != tuple(buckets) or min(buckets) < 1:
            raise ValueError("context-buckets must be positive, unique and increasing")
        if not modes or len(set(modes)) != len(modes) or set(modes) - set(MODES):
            raise ValueError("Invalid or duplicate modes")
        if not backends or len(set(backends)) != len(backends) or set(backends) - set(BACKENDS):
            raise ValueError("Invalid or duplicate backends")
        if "main_only" not in modes:
            raise ValueError("Include main_only as the matched baseline")
        self.model, self.output = model, Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        # A fresh directory prevents mixing records from different invocations.
        if any((self.output / name).exists() for name in ("calls.jsonl", "summary.csv", "metadata.json")):
            raise ValueError("Output already contains timing results; choose a new --output")
        self.warmup_calls, self.measure_calls = warmup_calls, measure_calls
        self.buckets, self.modes, self.backends = tuple(buckets), tuple(modes), tuple(backends)
        self.profile_trace, self.component_timings = profile_trace, component_timings
        self.rng = random.Random(seed)
        self.seen, self.rows, self.setup_rows, self.components = 0, [], [], []
        self.key, self.snapshot, self.graphs = None, None, {}
        self.diagnosed_keys = set()
        self.device = model.embed_tokens.weight.device
        if self.device.type != "cuda":
            raise ValueError("CUDA GPU required; CPU timings cannot answer stream overlap")
        model._side_stream = torch.cuda.Stream(device=self.device)
        self.metadata = dict(status="running", warmup_calls=warmup_calls,
            measure_calls=measure_calls, modes=self.modes, backends=self.backends,
            context_buckets=self.buckets, side_loop_steps=model.side_loop_steps,
            batch=1, block_size=model.block_size, seed=seed,
            device=torch.cuda.get_device_name(self.device), torch_version=torch.__version__,
            reference="main_only eager on original unpadded context cache",
            scope="real-input incremental draft forward: embedding, new-context projection, "
                  "cache concatenation, backbone, norm, head and greedy token selection; "
                  "target verification and static-input copies excluded",
            padding="old and new KV independently bucketed and masked, identical for both backends",
            padding_validation="elementwise close or sparse bounded BF16 tail, plus exact greedy tokens; "
                               "per-state metrics in padding_validation.jsonl",
            note="host_submit_ms includes Python and driver dispatch, not pure CPU compute; "
                 "GPU span can include dispatch stalls; profiler/components are separate calls")
        self.flush()

    @property
    def done(self):
        return self.seen >= self.warmup_calls + self.measure_calls

    def _setup(self, key):
        self.graphs.clear()
        self.snapshot = None
        torch.cuda.synchronize(self.device)
        setup_begin = time.perf_counter()
        self.snapshot = StaticDraftInput(self.model, *key)
        self.key = key
        return setup_begin

    @torch.inference_mode()
    def observe(self, *, model, draft_input_ids, position_ids, past_key_values,
                target_hidden_states, start):
        if self.done:
            raise TimingComplete()
        if model is not self.model or not isinstance(past_key_values, ContextKVCache):
            raise ValueError("Observer requires its own Qwen3 model and ContextKVCache")
        # Reject oversized prefixes even if old/new pieces individually fit.
        bucket_size(start, self.buckets)
        old, new = past_key_values.get_seq_length(), target_hidden_states.shape[1]
        # Decode adds at most one block of fresh target features. Do not pad
        # those 1..16 new tokens to a 1024-token context bucket.
        new_capacity = max(model.block_size, 1 << (new - 1).bit_length())
        key = (bucket_size(old, self.buckets), new_capacity)
        changed = key != self.key
        setup_begin = self._setup(key) if changed else None
        torch.cuda.synchronize(self.device)
        copy_begin = time.perf_counter()
        self.snapshot.update(draft_input_ids=draft_input_ids, position_ids=position_ids,
            target_hidden_states=target_hidden_states, past_key_values=past_key_values, start=start)
        torch.cuda.synchronize(self.device)
        copy_ms = (time.perf_counter() - copy_begin) * 1000
        # Graph capture and per-bucket validation are always outside measurements.
        if changed:
            for mode in self.modes:
                expected_hidden, expected_ids = self.snapshot.forward(mode)
                torch.cuda.synchronize(self.device)
                if "cudagraph" in self.backends:
                    try:
                        graph = GraphForward(self.snapshot, mode)
                    except Exception as exc:
                        raise RuntimeError(f"CUDA Graph capture failed for {mode}, buckets={key}; "
                            "no eager fallback was substituted. Try --backends eager to diagnose.") from exc
                    hidden, ids = graph()
                    torch.cuda.synchronize(self.device)
                    torch.testing.assert_close(hidden, expected_hidden, rtol=2e-2, atol=2e-2)
                    if not torch.equal(ids, expected_ids):
                        raise RuntimeError(f"Eager/Graph token mismatch for {mode}")
                    self.graphs[mode] = graph
            self.setup_rows.append(dict(call=self.seen + 1, old_bucket=key[0],
                new_bucket=key[1], setup_ms=(time.perf_counter() - setup_begin) * 1000))
        # Check masked snapshot against original unpadded incremental execution.
        # Only main_only is used to advance the reference decode trajectory.
        from deepspec.modeling.dspark.side_loop import prepare_paths
        isolated = ContextKVCache()
        isolated.layers = dict(past_key_values.layers)  # updates replace, never modify tensors in place
        prepared = prepare_paths(model, position_ids=position_ids,
            noise_embedding=model.embed_tokens(draft_input_ids),
            target_hidden_states=target_hidden_states, past_key_values=isolated, is_causal=False)
        expected, _ = run_paths(model, prepared, execution="main_only")
        actual, actual_tokens = self.snapshot.forward("main_only")
        torch.cuda.synchronize(self.device)
        report = padding_hidden_report(actual, expected)
        report.update(reference_call=self.seen + 1, context_tokens=start,
                      cached_tokens=old, new_context_tokens=new,
                      old_bucket=key[0], new_bucket=key[1])
        if report["passed"]:
            expected_tokens = model.compute_logits(expected[:, 1:]).argmax(-1)
            report["token_mismatches"] = int((actual_tokens != expected_tokens).sum().item())
            report["passed"] = report["token_mismatches"] == 0
            if not report["passed"]:
                report["status"] = "greedy_token_mismatch"
        with (self.output / "padding_validation.jsonl").open("a") as f:
            f.write(json.dumps(report) + "\n")
        if not report["passed"]:
            raise RuntimeError("Padded/unpadded forward validation failed: " + json.dumps(report))
        self.seen += 1
        warmup = self.seen <= self.warmup_calls
        groups = [(mode, backend) for mode in self.modes for backend in self.backends]
        self.rng.shuffle(groups)
        outputs = {}
        for mode, backend in groups:
            fn = self.graphs[mode] if backend == "cudagraph" else lambda m=mode: self.snapshot.forward(m)
            timing, (_, ids) = measure_forward(fn, self.device)
            outputs[(mode, backend)] = ids.clone()
            if not warmup:
                row = dict(call=self.seen - self.warmup_calls, reference_call=self.seen,
                    mode=mode, backend=backend, context_tokens=start, cached_tokens=old,
                    new_context_tokens=new, old_bucket=key[0], new_bucket=key[1],
                    input_copy_ms=copy_ms, **timing)
                self.rows.append(row)
                with (self.output / "calls.jsonl").open("a") as f:
                    f.write(json.dumps(row) + "\n")
        if "eager" in self.backends and "cudagraph" in self.backends:
            for mode in self.modes:
                if not torch.equal(outputs[(mode, "eager")], outputs[(mode, "cudagraph")]):
                    raise RuntimeError(f"Eager/Graph token mismatch at call {self.seen}, mode={mode}")
        if not warmup and key not in self.diagnosed_keys:
            self._diagnose(key)
            self.diagnosed_keys.add(key)
        self.flush()
        print(f"draft timing: warmup {min(self.seen, self.warmup_calls)}/{self.warmup_calls}, "
              f"measured {max(0, self.seen-self.warmup_calls)}/{self.measure_calls}, context={start}", flush=True)
        if self.done:
            raise TimingComplete()

    def _diagnose(self, key):
        if self.component_timings:
            for mode in self.modes:
                timer = PathTimer(True, self.device)
                self.snapshot.forward(mode, timer)
                torch.cuda.synchronize(self.device)
                self.components.append(dict(reference_call=self.seen, mode=mode,
                    old_bucket=key[0], new_bucket=key[1], backend="eager_instrumented",
                    **timer.resolve()))
        if self.profile_trace and "dual_stream" in self.modes:
            for backend in self.backends:
                fn = (self.graphs["dual_stream"] if backend == "cudagraph"
                      else lambda: self.snapshot.forward("dual_stream"))
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                     torch.profiler.ProfilerActivity.CUDA], record_shapes=True) as profile:
                    with torch.profiler.record_function(f"dual_stream/{backend}"):
                        fn()
                    torch.cuda.synchronize(self.device)
                profile.export_chrome_trace(str(self.output /
                    f"dual_stream_{backend}_old{key[0]}_new{key[1]}.trace.json"))

    def flush(self, *, status=None):
        if status is not None:
            self.metadata["status"] = status
        self.metadata.update(reference_calls_seen=self.seen,
            measured_calls_per_group=max(0, self.seen-self.warmup_calls),
            graph_setup=self.setup_rows)
        (self.output / "metadata.json").write_text(json.dumps(self.metadata, indent=2))
        (self.output / "components.json").write_text(json.dumps(self.components, indent=2))
        summaries = summarize_calls(self.rows)
        if summaries:
            with (self.output / "summary.csv").open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(summaries[0]))
                writer.writeheader()
                writer.writerows(summaries)
        return summaries

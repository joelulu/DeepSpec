"""Block-level adaptation of Ouro's two-stage exit objectives."""
import math

import torch
import torch.distributed as dist
import torch.nn.functional as F

from deepspec.utils.metrics import add_metric


def validate_exit_config(config):
    mode = getattr(config, "ouro_training_mode", "baseline")
    if mode not in ("baseline", "gate_only", "joint"):
        raise ValueError("ouro_training_mode must be baseline|gate_only|joint")
    if mode != "baseline" and not getattr(config, "enable_exit_gate", False):
        raise ValueError("Ouro training requires enable_exit_gate")
    if getattr(config, "enable_exit_gate", False):
        minimum = int(config.min_exit_loop)
        if not 1 <= minimum < int(config.num_loops):
            raise ValueError("min_exit_loop must be >= 1 and below num_loops")
    for name, default in (("ouro_beta", 0.1), ("gate_gain_slope", 50.0),
                          ("gate_gain_threshold", 0.005)):
        value = float(getattr(config, name, default))
        if not math.isfinite(value) or value < 0 or (name == "gate_gain_slope" and value == 0):
            raise ValueError(f"Invalid {name}: {value}")


def exit_distribution(stop_logits):
    """Nonfinal hazards [..., R-1] -> normalized exit PDF [..., R]."""
    remaining = torch.ones_like(stop_logits[..., 0], dtype=torch.float32)
    pdf = []
    for hazard in stop_logits.float().sigmoid().unbind(-1):
        pdf.append(remaining * hazard)
        remaining = remaining * (1 - hazard)
    pdf.append(remaining)
    return torch.stack(pdf, -1)


def block_cross_entropy(output):
    logits = output.draft_logits.float()
    ce = F.cross_entropy(logits.flatten(0, 2), output.target_ids.flatten(),
                         reduction="none").reshape_as(output.eval_mask)
    mask = output.eval_mask.float()
    counts = mask.sum(-1)
    valid = output.block_keep_mask.bool() & (counts > 0)
    return (ce * mask).sum(-1) / counts.clamp_min(1), valid


def valid_block_mean(values, valid):
    """Global valid-block mean with DDP/FSDP gradient averaging accounted for."""
    count = valid.float().sum().detach()
    world = dist.get_world_size() if dist.is_initialized() else 1
    if world > 1:
        dist.all_reduce(count)
    return (values * valid.float()).sum() * world / count.clamp_min(1)


def compute_ouro_loss(outputs, config, metric_tag="train"):
    mode = config.ouro_training_mode
    ces, valid = zip(*(block_cross_entropy(o) for o in outputs))
    valid = torch.stack(valid).all(0)
    for r, ce in enumerate(ces, 1):
        add_metric("ce_unweighted", (ce * valid).sum(), den=valid.sum(),
                   tag=f"{metric_tag}/loop{r}")
    start = int(config.min_exit_loop) - 1
    losses = torch.stack(ces[start:], -1)
    logits = torch.stack([o.exit_gate_logits for o in outputs[start:-1]], -1)
    add_metric("stop_prediction", (logits.float().sigmoid().mean(-1) * valid).sum(),
               den=valid.sum(), tag=metric_tag)
    if mode == "joint":
        pdf = exit_distribution(logits)
        entropy = -(pdf * pdf.clamp_min(1e-8).log()).sum(-1)
        expected = (pdf * losses).sum(-1)
        per_block = expected - float(config.ouro_beta) * entropy
        add_metric("exit_entropy", (entropy * valid).sum(), den=valid.sum(), tag=metric_tag)
        for index in range(pdf.shape[-1]):
            add_metric(f"exit_pdf/loop{start + index + 1}", (pdf[..., index] * valid).sum(),
                       den=valid.sum(), tag=metric_tag)
    elif mode == "gate_only":
        gain = (losses[..., :-1] - losses[..., 1:]).detach().clamp_min(0)
        target = 1 - torch.sigmoid(float(config.gate_gain_slope) *
                                  (gain - float(config.gate_gain_threshold)))
        per_block = F.binary_cross_entropy_with_logits(logits.float(), target, reduction="none").mean(-1)
        add_metric("stop_target", (target.mean(-1) * valid).sum(), den=valid.sum(), tag=metric_tag)
    else:
        raise ValueError("compute_ouro_loss requires gate_only or joint")
    return valid_block_mean(per_block, valid)

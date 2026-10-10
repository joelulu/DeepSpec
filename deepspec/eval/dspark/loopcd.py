"""Inference-only LoopCD readout for a frozen looped DFlash drafter."""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch


@dataclass(frozen=True)
class LoopCDConfig:
    early_loop: int = 1  # One-based exit index.
    strength: float = 0.2
    alpha: float = 0.1

    def __post_init__(self):
        if self.early_loop < 1:
            raise ValueError("LoopCD early_loop must be positive (one-based)")
        if not math.isfinite(self.strength) or self.strength < 0:
            raise ValueError("LoopCD lambda must be finite and nonnegative")
        if not math.isfinite(self.alpha) or not 0 < self.alpha <= 1:
            raise ValueError("LoopCD alpha must be in (0, 1]")

    def validate_model(self, model, num_loops: int) -> None:
        if not 1 <= self.early_loop < num_loops:
            raise ValueError("LoopCD requires 1 <= early_loop < evaluation num_loops")
        if model.markov_head is not None or model.confidence_head is not None:
            raise ValueError("LoopCD currently supports DFlash without Markov/confidence heads")


def loopcd_logits(
    last_logits: torch.Tensor,
    early_logits: torch.Tensor,
    config: LoopCDConfig,
) -> torch.Tensor:
    """log p_last - lambda log p_early, restricted by p_last plausibility.

    Log-softmax constants cancel for both argmax and renormalized sampling.
    Mask using *unmodified* final logits, before applying sampling temperature.
    The original final argmax always survives the mask.
    """
    if last_logits.shape != early_logits.shape:
        raise ValueError("LoopCD exit logits must have identical shapes")
    last, early = last_logits.float(), early_logits.float()
    keep = last >= last.amax(dim=-1, keepdim=True) + math.log(config.alpha)
    scores = last - config.strength * early
    return scores.masked_fill(~keep, float("-inf"))

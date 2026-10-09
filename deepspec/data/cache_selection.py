"""Deterministic logical cache views; never rewrite the cache or its index."""
from decimal import Decimal, InvalidOperation
import hashlib
import json
import random

import numpy as np


def select_cache_ids(total, percent=100, seed=42, holdout_samples=0, split="train"):
    try:
        percent = Decimal(str(percent))
    except InvalidOperation as exc:
        raise ValueError("data_percent must be in (0, 100]") from exc
    if not percent.is_finite() or not 0 < percent <= 100:
        raise ValueError("data_percent must be in (0, 100]")
    if total < 1 or not 0 <= holdout_samples < total:
        raise ValueError("holdout_samples must be nonnegative and smaller than the cache")
    if split not in ("train", "validation"):
        raise ValueError("split must be train or validation")
    held = np.array(sorted(random.Random(int(seed) + 104729).sample(range(total), holdout_samples)), dtype=np.int64)
    if split == "validation":
        if not len(held):
            raise ValueError("Validation requires holdout_samples > 0")
        return held
    count = int(Decimal(total - holdout_samples) * percent / 100)
    if not count:
        raise ValueError("Selected percentage contains no samples")
    if percent == 100 and not holdout_samples:
        return None  # Preserve the original full-cache behavior and avoid an allocation.
    pool = np.delete(np.arange(total, dtype=np.int64), held)
    if count == len(pool):
        return pool
    indices = sorted(random.Random(int(seed)).sample(range(len(pool)), count))
    return pool[indices]


def selection_identity(cache_dir, manifest, ids, *, percent, seed, holdout_samples, split):
    payload = dict(cache_dir=cache_dir, manifest=manifest, data_percent=str(Decimal(str(percent)).normalize()),
                   seed=int(seed), holdout_samples=int(holdout_samples), split=split)
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode())
    if ids is not None:
        digest.update(ids.astype("<i8", copy=False).tobytes())
    return digest.hexdigest()

"""Persistent target-context KV only; draft-block KV never enters this cache."""
from __future__ import annotations

import torch


class ContextKVCache:
    def __init__(self):
        self.layers: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}

    def get_seq_length(self, layer_idx: int = 0) -> int:
        pair = self.layers.get(layer_idx)
        return 0 if pair is None else pair[0].shape[-2]

    def update(self, key, value, layer_idx: int):
        if layer_idx in self.layers:
            old_key, old_value = self.layers[layer_idx]
            key = torch.cat((old_key, key), dim=-2)
            value = torch.cat((old_value, value), dim=-2)
        self.layers[layer_idx] = key, value
        return key, value

    def crop(self, length: int):
        if length < 0:
            raise ValueError("ContextKVCache.crop requires an absolute length")
        for index, (key, value) in self.layers.items():
            self.layers[index] = key[..., :length, :], value[..., :length, :]

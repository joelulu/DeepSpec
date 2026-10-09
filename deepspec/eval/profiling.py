"""Optional stage GPU-event timing and synchronized decode wall time."""
import time
from contextlib import contextmanager

import torch


class DecodeProfiler:
    def __init__(self, device, enabled=False):
        self.device = device
        self.enabled = enabled
        self.events = {name: [] for name in ("prefill", "draft", "verify")}
        self.decode_start = None
        if self.enabled and self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self.generation_start = time.perf_counter() if self.enabled else None

    @contextmanager
    def stage(self, name):
        if not self.enabled:
            yield
            return
        if self.device.type == "cuda":
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            yield
            end.record()
        else:
            start = time.perf_counter()
            yield
            end = time.perf_counter()
        self.events[name].append((start, end))

    def start_decode(self):
        if self.enabled:
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            self.decode_start = time.perf_counter()

    def finish(self):
        if not self.enabled:
            return {}
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        elapsed = 0.0 if self.decode_start is None else (time.perf_counter() - self.decode_start) * 1000
        result = {"decode_wall_ms": elapsed,
                  "generation_wall_ms": (time.perf_counter() - self.generation_start) * 1000}
        for name, pairs in self.events.items():
            result[f"{name}_ms"] = sum(
                start.elapsed_time(end) if self.device.type == "cuda" else (end - start) * 1000
                for start, end in pairs
            )
        return result

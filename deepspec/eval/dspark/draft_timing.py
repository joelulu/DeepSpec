"""Matched-input forward timing on states from real speculative decoding.

Only plain Qwen3 DFlash fusion checkpoints are supported. Graphs use exact-shape
projected-context inputs and fixed buffers, recaptured when shapes change. No diagnostic forward writes
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
        base = next((r for r in summaries if r["mode"] == "main_only"
                    and r["backend"] == row["backend"]), None)
        # Preserve the original error if a diagnostic fails partway through a
        # shuffled group list before its main-only baseline has been measured.
        row["extra_wall_ms"] = row["wall_ms_mean"] - base["wall_ms_mean"] if base else None
        row["overhead_wall_pct"] = 100 * row["extra_wall_ms"] / base["wall_ms_mean"] if base else None
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
    """Fixed-address snapshot with exactly the reference forward's shapes.

    No padding or extra attention mask is introduced. A different (old, new)
    context length needs a new snapshot and graph capture. Input copies are
    outside forward timing and reported separately; projection of new features
    and cache concatenation are measured. Diagnostic calls never mutate the
    reference cache.
    """
    def __init__(self, model, old_capacity, new_capacity):
        self.model = model
        self.old_capacity, self.new_capacity = old_capacity, new_capacity
        weight = model.embed_tokens.weight
        device, dtype = weight.device, weight.dtype
        self.device = device
        self.ids = torch.empty((1, model.block_size), device=device, dtype=torch.long)
        self.features = torch.empty((1, new_capacity,
            len(model.target_layer_ids) * model.config.hidden_size), device=device, dtype=dtype)
        self.positions = torch.empty((1, new_capacity + model.block_size),
                                     device=device, dtype=torch.long)
        self.old_kv = []
        for layer in model.layers:
            attn = layer.self_attn
            shape = (1, attn.num_key_value_heads, old_capacity, attn.head_dim)
            self.old_kv.append((torch.empty(shape, device=device, dtype=dtype),
                                torch.empty(shape, device=device, dtype=dtype)))

    def update(self, *, draft_input_ids, target_hidden_states, past_key_values,
               position_ids, start):
        old = past_key_values.get_seq_length()
        new = target_hidden_states.shape[1]
        if old + new != start:
            raise ValueError(f"Context alignment: cached={old}, new={new}, anchor={start}")
        if (old, new) != (self.old_capacity, self.new_capacity):
            raise ValueError("Exact-shape snapshot requires recapture for changed context lengths")
        if draft_input_ids.shape != self.ids.shape or position_ids.shape != self.positions.shape:
            raise ValueError("Expected one complete draft block and aligned position IDs")
        self.ids.copy_(draft_input_ids)
        self.features.copy_(target_hidden_states)
        self.positions.copy_(position_ids)
        for index, (key, value) in enumerate(self.old_kv):
            if old:
                if index not in past_key_values.layers:
                    raise ValueError(f"Missing reference context KV at layer {index}")
                old_key, old_value = past_key_values.layers[index]
                if old_key.shape != key.shape or old_value.shape != value.shape:
                    raise ValueError("Reference cache layers have unequal shapes")
                key.copy_(old_key)
                value.copy_(old_value)

    def forward(self, mode, timer=None):
        if mode not in MODES:
            raise ValueError(f"Unknown timing mode {mode}")
        model = self.model
        timer = timer or PathTimer(False, self.device)
        with timer.stage("prepare"):
            noise = model.embed_tokens(self.ids)
            context = model.hidden_norm(model.fc(self.features))
            rope = model.rotary_emb(noise, self.positions)
            projected = [None] * 5
            needed = (0,) if mode == "side_only" else range(5)
            for index in needed:
                key, value = model.layers[index].self_attn.project_context(context, rope)
                if self.old_capacity:
                    old_key, old_value = self.old_kv[index]
                    key = torch.cat((old_key, key), dim=-2)
                    value = torch.cat((old_value, value), dim=-2)
                # Match ContextKVCache.update: no concatenation for an empty
                # cache. Avoid changing first-call KV layout/strides.
                projected[index] = key, value
            prepared = dict(context=context, rope=rope, projected=projected,
                position_ids=self.positions, attention_mask=None, noise=noise,
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
            input_layout="exact original context/new-feature lengths, no padding or extra mask",
            graph_policy="recapture all selected modes when (cached_tokens, new_tokens) changes; "
                         "setup excluded from measured forward latency",
            validation="original elementwise hidden tolerance and exact greedy tokens; "
                       "metrics in input_validation.jsonl",
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
        # --context-buckets remains a compatible upper-limit/diagnostic grouping
        # argument. It no longer changes any tensor shape.
        diagnostic_bucket = bucket_size(start, self.buckets)
        old, new = past_key_values.get_seq_length(), target_hidden_states.shape[1]
        key = (old, new)
        changed = key != self.key
        setup_begin = self._setup(key) if changed else None
        torch.cuda.synchronize(self.device)
        copy_begin = time.perf_counter()
        self.snapshot.update(draft_input_ids=draft_input_ids, position_ids=position_ids,
            target_hidden_states=target_hidden_states, past_key_values=past_key_values, start=start)
        torch.cuda.synchronize(self.device)
        copy_ms = (time.perf_counter() - copy_begin) * 1000
        # Exact-shape graph capture/validation are always outside measurements.
        if changed:
            for mode in self.modes:
                expected_hidden, expected_ids = self.snapshot.forward(mode)
                torch.cuda.synchronize(self.device)
                if "cudagraph" in self.backends:
                    try:
                        graph = GraphForward(self.snapshot, mode)
                    except Exception as exc:
                        raise RuntimeError(f"CUDA Graph capture failed for {mode}, exact_shape={key}; "
                            "no eager fallback was substituted. Try --backends eager to diagnose.") from exc
                    hidden, ids = graph()
                    torch.cuda.synchronize(self.device)
                    torch.testing.assert_close(hidden, expected_hidden, rtol=2e-2, atol=2e-2)
                    if not torch.equal(ids, expected_ids):
                        raise RuntimeError(f"Eager/Graph token mismatch for {mode}")
                    self.graphs[mode] = graph
            self.setup_rows.append(dict(call=self.seen + 1, cached_tokens=key[0],
                new_context_tokens=key[1], setup_ms=(time.perf_counter() - setup_begin) * 1000))
        # Check an exact-shape snapshot against the original incremental path.
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
        report = dict(reference_call=self.seen + 1, context_tokens=start,
            cached_tokens=old, new_context_tokens=new, input_layout="exact", passed=False)
        try:
            torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)
            expected_tokens = model.compute_logits(expected[:, 1:]).argmax(-1)
            report["token_mismatches"] = int((actual_tokens != expected_tokens).sum().item())
            if report["token_mismatches"]:
                raise RuntimeError("Exact-shape snapshot changed main-only greedy tokens")
            report["passed"] = True
        except (AssertionError, RuntimeError) as exc:
            report["error"] = str(exc)
            raise
        finally:
            report["max_abs"] = float((actual.float() - expected.float()).abs().max().item())
            with (self.output / "input_validation.jsonl").open("a") as f:
                f.write(json.dumps(report) + "\n")
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
                    new_context_tokens=new, input_layout="exact",
                    input_copy_ms=copy_ms, **timing)
                self.rows.append(row)
                with (self.output / "calls.jsonl").open("a") as f:
                    f.write(json.dumps(row) + "\n")
        if "eager" in self.backends and "cudagraph" in self.backends:
            for mode in self.modes:
                if not torch.equal(outputs[(mode, "eager")], outputs[(mode, "cudagraph")]):
                    raise RuntimeError(f"Eager/Graph token mismatch at call {self.seen}, mode={mode}")
        # Exact shapes usually change every round. Capture diagnostic traces
        # only once per context range, not 50 sets of traces/components.
        if not warmup and diagnostic_bucket not in self.diagnosed_keys:
            self._diagnose(key)
            self.diagnosed_keys.add(diagnostic_bucket)
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
                    cached_tokens=key[0], new_context_tokens=key[1], backend="eager_instrumented",
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
                    f"dual_stream_{backend}_cached{key[0]}_new{key[1]}.trace.json"))

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

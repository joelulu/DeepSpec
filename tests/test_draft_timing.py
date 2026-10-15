"""Verify matched snapshots, cache isolation and the loop-sweep timing protocol."""
import json
from types import SimpleNamespace

import pytest
import torch

from deepspec.eval.dspark.draft_ops import forward_dspark_draft_block
from deepspec.eval.dspark.draft_timing import (MODES, DraftTimingObserver, GraphForward,
    StaticDraftInput, TimingComplete, bucket_size, summarize_calls)
from deepspec.modeling.dspark.context_cache import ContextKVCache
from deepspec.modeling.dspark.qwen3 import Qwen3DSparkModel
from deepspec.modeling.dspark.side_loop import prepare_paths, run_paths
from scripts.side.benchmark_inference import parse_args
from test_side_loop import side_config


def model_and_ids(device="cpu"):
    torch.manual_seed(42)
    torch.set_num_threads(1)
    cfg = side_config("fusion", block=16)
    cfg.sample_from_anchor = False
    model = Qwen3DSparkModel(cfg).to(device).eval()
    with torch.no_grad():
        model.side_fusion.up.weight.fill_(0.04)
    ids = torch.full((1, 16), model.mask_token_id, device=device)
    ids[:, 0] = 2
    return model, ids


@pytest.mark.parametrize("mode", ["main_only", "side_only", "fusion_only", "serial_dual"])
def test_exact_snapshot_matches_original_forward_and_preserves_cache(mode):
    model, ids = model_and_ids()
    cache = ContextKVCache()
    with torch.inference_mode():
        for old, new in ((0, 6), (6, 2), (8, 1)):
            features = torch.randn(1, new, 64)
            positions = torch.arange(old, old + new + 16).unsqueeze(0)
            snapshot = StaticDraftInput(model, old, new)
            snapshot.update(draft_input_ids=ids, target_hidden_states=features,
                past_key_values=cache, position_ids=positions, start=old + new)
            before = {i: tuple(t.clone() for t in pair) for i, pair in cache.layers.items()}
            hooks, calls = [], [0] * 5
            for i, layer in enumerate(model.layers):
                def count(module, args, output, index=i):
                    calls[index] += 1
                hooks.append(layer.register_forward_hook(count))
            actual, tokens = snapshot.forward(mode)
            for hook in hooks:
                hook.remove()
            assert calls == ([5, 0, 0, 0, 0] if mode == "side_only"
                             else [5, 1, 1, 1, 1] if mode == "serial_dual" else [1, 1, 1, 1, 1])
            assert cache.get_seq_length() == old
            for i in before:
                for a, b in zip(before[i], cache.layers[i]):
                    torch.testing.assert_close(a, b, atol=0, rtol=0)
            isolated = ContextKVCache()
            isolated.layers = dict(cache.layers)
            prepared = prepare_paths(model, position_ids=positions,
                noise_embedding=model.embed_tokens(ids), target_hidden_states=features,
                past_key_values=isolated, is_causal=False)
            expected, _ = run_paths(model, prepared, execution=mode)
            torch.testing.assert_close(actual, expected, atol=3e-6, rtol=3e-5)
            assert torch.equal(tokens, model.compute_logits(expected[:, 1:]).argmax(-1))
            # Advance only the reference cache, exactly once.
            cache = isolated


def test_fusion_only_uses_h1_without_extra_recurrence():
    model, ids = model_and_ids()
    with torch.inference_mode():
        prepared = prepare_paths(model, position_ids=torch.arange(22).unsqueeze(0),
            noise_embedding=model.embed_tokens(ids), target_hidden_states=torch.randn(1, 6, 64))
        first = model.layers[0](hidden_states=prepared["noise"], target_hidden_states=prepared["context"],
            position_embeddings=prepared["rope"], position_ids=prepared["position_ids"],
            attention_mask=None, projected_context_kv=prepared["projected"][0])
        main = first
        for i in range(1, 5):
            main = model.layers[i](hidden_states=main, target_hidden_states=prepared["context"],
                position_embeddings=prepared["rope"], position_ids=prepared["position_ids"],
                attention_mask=None, projected_context_kv=prepared["projected"][i])
        actual, _ = run_paths(model, prepared, execution="fusion_only")
        torch.testing.assert_close(actual, model.norm(model.side_fusion(main, first)))


def test_observer_hook_receives_cache_before_update_and_can_stop_without_mutation():
    model, ids = model_and_ids()
    cache = ContextKVCache()
    seen = []
    class Observer:
        def observe(self, **kwargs):
            seen.append((kwargs["start"], kwargs["past_key_values"].get_seq_length()))
            raise TimingComplete()
    model.draft_timing_observer = Observer()
    with torch.inference_mode(), pytest.raises(TimingComplete):
        forward_dspark_draft_block(model, draft_input_ids=ids,
            position_ids=torch.arange(22).unsqueeze(0), past_key_values_draft=cache,
            target_hidden_states=torch.randn(1, 6, 64), start=6, block_size=16)
    assert seen == [(6, 0)] and cache.layers == {}


def test_summary_divides_by_actual_calls_and_compares_matching_backend():
    rows = []
    for backend, base in (("eager", 4.0), ("cudagraph", 2.0)):
        for mode in MODES:
            for offset in (0.0, 1.0):
                value = base + offset + (1.0 if mode == "dual_stream" else 0.0)
                rows.append(dict(mode=mode, backend=backend, context_tokens=8,
                    wall_ms=value, gpu_ms=value - 0.1, host_submit_ms=0.2, input_copy_ms=0.3))
    summaries = summarize_calls(rows)
    assert len(summaries) == 10
    eager = next(r for r in summaries if r["mode"] == "dual_stream" and r["backend"] == "eager")
    graph = next(r for r in summaries if r["mode"] == "dual_stream" and r["backend"] == "cudagraph")
    assert eager["calls"] == 2 and eager["wall_ms_mean"] == 5.5
    assert eager["overhead_wall_pct"] == pytest.approx(100 / 4.5)
    assert graph["overhead_wall_pct"] == pytest.approx(100 / 2.5)


def test_cli_validation_and_buckets():
    args = parse_args(["--target", "target", "--draft", "draft", "--output", "out"])
    assert args.warmup_calls == 5 and args.measure_calls == 50
    assert len(args.modes) * len(args.backends) == 10
    assert args.loop_counts == (5,)
    assert bucket_size(0, (16, 32)) == 0
    assert bucket_size(17, (16, 32)) == 32
    with pytest.raises(ValueError, match="never truncated"):
        bucket_size(33, (16, 32))
    with pytest.raises(SystemExit):
        parse_args(["--target", "t", "--draft", "d", "--output", "o", "--measure-calls", "0"])


def test_partial_summary_does_not_hide_failure_when_baseline_is_not_yet_measured():
    row = dict(mode="dual_stream", backend="eager", context_tokens=96,
               wall_ms=4.0, gpu_ms=3.9, host_submit_ms=0.3, input_copy_ms=0.1)
    summary = summarize_calls([row])
    assert summary[0]["calls"] == 1
    assert summary[0]["overhead_wall_pct"] is None


def test_changed_context_shape_requires_new_snapshot():
    model, ids = model_and_ids()
    cache = ContextKVCache()
    snapshot = StaticDraftInput(model, 0, 6)
    with torch.inference_mode(), pytest.raises(ValueError, match="recapture"):
        snapshot.update(draft_input_ids=ids, target_hidden_states=torch.randn(1, 7, 64),
            past_key_values=cache, position_ids=torch.arange(23).unsqueeze(0), start=7)


@pytest.mark.parametrize("loops", [1, 2, 3, 4, 5])
@pytest.mark.parametrize("mode", ["main_only", "side_only", "fusion_only", "serial_dual"])
def test_loop_override_counts_shared_l1_once_and_preserves_model(mode, loops):
    model, ids = model_and_ids()
    snapshot = StaticDraftInput(model, 0, 6)
    with torch.inference_mode():
        snapshot.update(draft_input_ids=ids, target_hidden_states=torch.randn(1, 6, 64),
            past_key_values=ContextKVCache(), position_ids=torch.arange(22).unsqueeze(0), start=6)
        calls = [0] * 5
        hooks = []
        for i, layer in enumerate(model.layers):
            def count(module, args, output, index=i):
                calls[index] += 1
            hooks.append(layer.register_forward_hook(count))
        actual, tokens = snapshot.forward(mode, side_loops=loops)
        for hook in hooks:
            hook.remove()
        expected = ([loops, 0, 0, 0, 0] if mode == "side_only" else
                    [loops, 1, 1, 1, 1] if mode == "serial_dual" else [1, 1, 1, 1, 1])
        assert calls == expected
        assert model.side_loop_steps == 5
        if loops == 1 and mode == "serial_dual":
            fusion, fusion_tokens = snapshot.forward("fusion_only")
            torch.testing.assert_close(actual, fusion, atol=0, rtol=0)
            assert torch.equal(tokens, fusion_tokens)
        # An override must not leak into the next group or reference model.
        default, _ = snapshot.forward(mode)
        explicit, _ = snapshot.forward(mode, side_loops=5)
        torch.testing.assert_close(default, explicit, atol=0, rtol=0)


def test_sweep_summary_keeps_loops_and_backends_separate_and_reports_serial_saving():
    rows = []
    for loops in (1, 5):
        for backend, base in (("eager", 4.0), ("cudagraph", 2.0)):
            for mode, value in (("main_only", base), ("serial_dual", base + loops),
                                ("dual_stream", base + loops / 2)):
                rows.append(dict(loops=loops, mode=mode, backend=backend, context_tokens=100,
                    wall_ms=value, gpu_ms=value, host_submit_ms=0.1, input_copy_ms=0.2))
    summary = summarize_calls(rows)
    assert len(summary) == 12
    for row in summary:
        if row["mode"] == "dual_stream":
            base = 4.0 if row["backend"] == "eager" else 2.0
            loops = row["loops"]
            assert row["overhead_wall_pct"] == pytest.approx(100 * (loops / 2) / base)
            assert row["saved_vs_serial_wall_ms"] == loops / 2
            assert row["saved_vs_serial_wall_pct"] == pytest.approx(100 * (loops / 2) / (base + loops))
            assert row["speedup_vs_serial"] == pytest.approx((base + loops) / (base + loops / 2))
        else:
            assert row["saved_vs_serial_wall_ms"] is None


def test_cli_loop_sweep_overrides_single_count_and_accepts_serial():
    base = ["--target", "t", "--draft", "d", "--output", "o"]
    args = parse_args(base + ["--loop-sweep", "1,2,3,4,5", "--backends", "cudagraph"])
    assert args.loop_counts == (1, 2, 3, 4, 5)
    assert "serial_dual" in args.modes
    assert len(args.loop_counts) * len(args.modes) * len(args.backends) == 25
    for invalid in ("", "0,1", "1,1", "2,1", "x", "1,"):
        with pytest.raises(SystemExit):
            parse_args(base + ["--loop-sweep", invalid])


@pytest.mark.parametrize("new", [1, 2, 5, 6])
def test_bf16_exact_snapshot_keeps_original_projection_shapes(new):
    model, ids = model_and_ids()
    model = model.to(dtype=torch.bfloat16)
    cache = ContextKVCache()
    with torch.inference_mode():
        prefix = torch.randn(1, 6, 64, dtype=torch.bfloat16)
        prepare_paths(model, position_ids=torch.arange(22).unsqueeze(0),
            noise_embedding=model.embed_tokens(ids), target_hidden_states=prefix,
            past_key_values=cache, is_causal=False)
        features = torch.randn(1, new, 64, dtype=torch.bfloat16)
        positions = torch.arange(6, 6 + new + 16).unsqueeze(0)
        snapshot = StaticDraftInput(model, 6, new)
        snapshot.update(draft_input_ids=ids, target_hidden_states=features,
            past_key_values=cache, position_ids=positions, start=6 + new)
        shapes = []
        hook = model.fc.register_forward_pre_hook(lambda module, args: shapes.append(args[0].shape))
        actual, _ = snapshot.forward("main_only")
        isolated = ContextKVCache()
        isolated.layers = dict(cache.layers)
        prepared = prepare_paths(model, position_ids=positions,
            noise_embedding=model.embed_tokens(ids), target_hidden_states=features,
            past_key_values=isolated, is_causal=False)
        expected, _ = run_paths(model, prepared, execution="main_only")
        hook.remove()
        assert shapes == [torch.Size([1, new, 64])] * 2
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA Graph/overlap requires GPU")
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("loops", [1, 3, 5])
def test_cuda_exact_graph_replays_updated_inputs_and_matches_eager(mode, loops):
    model, ids = model_and_ids("cuda")
    model = model.to(dtype=torch.bfloat16)
    model._side_stream = torch.cuda.Stream()
    cache = ContextKVCache()
    with torch.inference_mode():
        prepare_paths(model, position_ids=torch.arange(22, device="cuda").unsqueeze(0),
            noise_embedding=model.embed_tokens(ids), target_hidden_states=torch.randn(
                1, 6, 64, device="cuda", dtype=torch.bfloat16), past_key_values=cache)
        snapshot = StaticDraftInput(model, 6, 2)
        features = torch.randn(1, 2, 64, device="cuda", dtype=torch.bfloat16)
        positions = torch.arange(6, 24, device="cuda").unsqueeze(0)
        snapshot.update(draft_input_ids=ids, target_hidden_states=features,
            past_key_values=cache, position_ids=positions, start=8)
        graph = GraphForward(snapshot, mode, side_loops=loops)
        for iteration in range(2):
            new = 2
            ids[:, 0] = 3 + iteration
            snapshot.update(draft_input_ids=ids, target_hidden_states=torch.randn(
                1, new, 64, device="cuda", dtype=torch.bfloat16), past_key_values=cache,
                position_ids=positions, start=6 + new)
            expected, expected_ids = snapshot.forward(mode, side_loops=loops)
            actual, actual_ids = graph()
            torch.cuda.synchronize()
            torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)
            assert torch.equal(actual_ids, expected_ids)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Timing protocol requires GPU")
def test_cuda_observer_excludes_warmup_stops_at_budget_and_keeps_reference_cache(tmp_path):
    model, ids = model_and_ids("cuda")
    model = model.to(dtype=torch.bfloat16)
    observer = DraftTimingObserver(model, tmp_path, warmup_calls=1, measure_calls=2,
        buckets=(16, 32), component_timings=False, loop_counts=(1, 3, 5))
    cache = ContextKVCache()
    with torch.inference_mode():
        kwargs = dict(model=model, draft_input_ids=ids,
            target_hidden_states=torch.randn(1, 6, 64, device="cuda", dtype=torch.bfloat16),
            past_key_values=cache, position_ids=torch.arange(22, device="cuda").unsqueeze(0), start=6)
        for old, new in ((0, 6), (6, 2)):
            kwargs.update(target_hidden_states=torch.randn(1, new, 64, device="cuda", dtype=torch.bfloat16),
                position_ids=torch.arange(old, old + new + 16, device="cuda").unsqueeze(0), start=old + new)
            observer.observe(**kwargs)
            if old == 0:
                assert not observer.rows
            prepare_paths(model, position_ids=kwargs["position_ids"], noise_embedding=model.embed_tokens(ids),
                target_hidden_states=kwargs["target_hidden_states"], past_key_values=cache)
        before = {i: tuple(t.clone() for t in pair) for i, pair in cache.layers.items()}
        kwargs.update(target_hidden_states=torch.randn(1, 2, 64, device="cuda", dtype=torch.bfloat16),
            position_ids=torch.arange(8, 26, device="cuda").unsqueeze(0), start=10)
        with pytest.raises(TimingComplete):
            observer.observe(**kwargs)
    assert cache.get_seq_length() == 8
    for i in before:
        for a, b in zip(before[i], cache.layers[i]):
            torch.testing.assert_close(a, b, atol=0, rtol=0)
    assert len(observer.setup_rows) == 3
    summary = observer.flush(status="complete")
    assert len(summary) == 30 and all(r["calls"] == 2 for r in summary)
    records = [json.loads(line) for line in (tmp_path / "calls.jsonl").read_text().splitlines()]
    assert len(records) == 60
    assert {r["reference_call"] for r in records} == {2, 3}
    for setup in observer.setup_rows:
        assert setup["serial_dual_validated_loops"] == [1, 3, 5]
    assert json.loads((tmp_path / "metadata.json").read_text())["status"] == "complete"

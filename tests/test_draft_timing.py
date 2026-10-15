"""Verify matched snapshots, cache isolation and the eight-way timing protocol."""
import json
from types import SimpleNamespace

import pytest
import torch

from deepspec.eval.dspark.draft_ops import forward_dspark_draft_block
from deepspec.eval.dspark.draft_timing import (MODES, DraftTimingObserver, GraphForward,
    StaticDraftInput, TimingComplete, bucket_size, padding_hidden_report, summarize_calls)
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


@pytest.mark.parametrize("mode", ["main_only", "side_only", "fusion_only"])
def test_padded_incremental_snapshot_matches_unpadded_forward_and_preserves_cache(mode):
    model, ids = model_and_ids()
    cache = ContextKVCache()
    with torch.inference_mode():
        for old, new in ((0, 6), (6, 2), (8, 1)):
            features = torch.randn(1, new, 64)
            positions = torch.arange(old, old + new + 16).unsqueeze(0)
            snapshot = StaticDraftInput(model, 0 if old == 0 else 16, 16)
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
            assert calls == ([5, 0, 0, 0, 0] if mode == "side_only" else [1, 1, 1, 1, 1])
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
    assert len(summaries) == 8
    eager = next(r for r in summaries if r["mode"] == "dual_stream" and r["backend"] == "eager")
    graph = next(r for r in summaries if r["mode"] == "dual_stream" and r["backend"] == "cudagraph")
    assert eager["calls"] == 2 and eager["wall_ms_mean"] == 5.5
    assert eager["overhead_wall_pct"] == pytest.approx(100 / 4.5)
    assert graph["overhead_wall_pct"] == pytest.approx(100 / 2.5)


def test_cli_validation_and_buckets():
    args = parse_args(["--target", "target", "--draft", "draft", "--output", "out"])
    assert args.warmup_calls == 5 and args.measure_calls == 50
    assert len(args.modes) * len(args.backends) == 8
    assert bucket_size(0, (16, 32)) == 0
    assert bucket_size(17, (16, 32)) == 32
    with pytest.raises(ValueError, match="never truncated"):
        bucket_size(33, (16, 32))
    with pytest.raises(SystemExit):
        parse_args(["--target", "t", "--draft", "d", "--output", "o", "--measure-calls", "0"])


def test_bf16_sparse_tail_does_not_fail_on_one_element_of_40960():
    expected = torch.ones(1, 16, 2560, dtype=torch.bfloat16)
    expected[0, 1, 67] = 0.27
    actual = expected.clone()
    actual[0, 1, 67] += 0.0302734375
    with pytest.raises(AssertionError):
        torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.02)
    report = padding_hidden_report(actual, expected)
    assert report["passed"] and report["status"] == "bounded_bf16_tail"
    assert report["elementwise_mismatches"] == 1 and report["elements"] == 40960
    assert report["relative_l2"] < 0.001


@pytest.mark.parametrize("kind", ["widespread", "single_large", "one_position", "nonfinite"])
def test_padding_validation_still_rejects_substantial_error(kind):
    expected = torch.ones(1, 16, 2560, dtype=torch.bfloat16)
    actual = expected.clone()
    if kind == "widespread":
        actual += 0.05
    elif kind == "single_large":
        actual[0, 1, 67] += 2.0
    elif kind == "one_position":
        actual[0, 1] += 0.05
    else:
        actual[0, 1, 67] = float("nan")
    assert not padding_hidden_report(actual, expected)["passed"]


def test_sparse_tail_allowance_is_not_used_for_fp32():
    expected = torch.ones(1, 16, 2560)
    expected[0, 1, 67] = 0.27
    actual = expected.clone()
    actual[0, 1, 67] += 0.03
    assert not padding_hidden_report(actual, expected)["passed"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA Graph/overlap requires GPU")
@pytest.mark.parametrize("mode", MODES)
def test_cuda_graph_replays_new_real_inputs_and_matches_eager(mode):
    model, ids = model_and_ids("cuda")
    model = model.to(dtype=torch.bfloat16)
    model._side_stream = torch.cuda.Stream()
    cache = ContextKVCache()
    snapshot = StaticDraftInput(model, 16, 16)
    with torch.inference_mode():
        features = torch.randn(1, 6, 64, device="cuda", dtype=torch.bfloat16)
        positions = torch.arange(22, device="cuda").unsqueeze(0)
        snapshot.update(draft_input_ids=ids, target_hidden_states=features,
            past_key_values=cache, position_ids=positions, start=6)
        graph = GraphForward(snapshot, mode)
        for new in (6, 9):
            ids[:, 0] = 3 + new
            snapshot.update(draft_input_ids=ids, target_hidden_states=torch.randn(
                1, new, 64, device="cuda", dtype=torch.bfloat16), past_key_values=cache,
                position_ids=torch.arange(new + 16, device="cuda").unsqueeze(0), start=new)
            expected, expected_ids = snapshot.forward(mode)
            actual, actual_ids = graph()
            torch.cuda.synchronize()
            torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)
            assert torch.equal(actual_ids, expected_ids)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Timing protocol requires GPU")
def test_cuda_observer_excludes_warmup_stops_at_budget_and_keeps_reference_cache(tmp_path):
    model, ids = model_and_ids("cuda")
    model = model.to(dtype=torch.bfloat16)
    observer = DraftTimingObserver(model, tmp_path, warmup_calls=1, measure_calls=2,
        buckets=(16, 32), component_timings=False)
    cache = ContextKVCache()
    with torch.inference_mode():
        kwargs = dict(model=model, draft_input_ids=ids,
            target_hidden_states=torch.randn(1, 6, 64, device="cuda", dtype=torch.bfloat16),
            past_key_values=cache, position_ids=torch.arange(22, device="cuda").unsqueeze(0), start=6)
        observer.observe(**kwargs)
        assert not observer.rows
        observer.observe(**kwargs)
        with pytest.raises(TimingComplete):
            observer.observe(**kwargs)
    assert cache.layers == {}
    summary = observer.flush(status="complete")
    assert len(summary) == 8 and all(r["calls"] == 2 for r in summary)
    assert len((tmp_path / "calls.jsonl").read_text().splitlines()) == 16
    assert json.loads((tmp_path / "metadata.json").read_text())["status"] == "complete"

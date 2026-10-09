import copy
import importlib.util
import itertools
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from transformers import DynamicCache, Qwen3ForCausalLM

from deepspec.data.target_cache_dataset import (CacheDataset, LocalTargetCacheWriter,
    build_target_cache_manifest, write_target_cache_manifest)
from deepspec.eval.dspark.depth_probe import probe_depths
from deepspec.modeling.dspark.common import extract_context_feature
from deepspec.modeling.dspark.context_cache import ContextKVCache
from deepspec.modeling.dspark.qwen3.modeling import Qwen3DSparkModel
from test_loop_qwen3 import config


def load_script(name):
    path = Path(__file__).resolve().parents[1] / "scripts" / "loop" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_subset_preserves_actual_tensor_data_and_reuses_identical_selection(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    writer = LocalTargetCacheWriter(rank_dir=str(source), max_shard_bytes=2048)
    for i in range(100):
        writer.write_sample(sample_id=i, input_ids=torch.tensor([i, i + 1]),
            attention_mask=torch.ones(2), loss_mask=torch.ones(2),
            target_hidden_states=torch.full((2, 4), i, dtype=torch.bfloat16),
            target_last_hidden_states=torch.full((2, 4), i + 1, dtype=torch.bfloat16))
    writer.close()
    (source / "samples.local.idx").rename(source / "samples.idx")
    manifest = build_target_cache_manifest(num_samples=100,
        shards=[dict(shard_id=i, file_name=name) for i, name in enumerate(writer.local_shard_files)],
        target_layer_ids=[0], hidden_size=4, extra_fields=dict(target_model_name_or_path="target"))
    write_target_cache_manifest(output_dir=str(source), manifest=manifest)
    script = load_script("cache_subset")
    before = script.digest(source / "samples.idx")
    dest = tmp_path / "subset"
    script.create_subset(source, dest)
    selected = json.loads((dest / "selected_source_ids.json").read_text())
    subset, original = CacheDataset(str(dest)), CacheDataset(str(source))
    assert len(subset) == 3
    for new_id, old_id in enumerate(selected):
        for key, value in original[old_id].items():
            torch.testing.assert_close(subset[new_id][key], value)
    subset.close(); original.close()
    assert script.digest(source / "samples.idx") == before
    index_before = (dest / "samples.idx").read_bytes()
    script.create_subset(source, dest)
    assert (dest / "samples.idx").read_bytes() == index_before
    assert all((dest / s["file_name"]).is_symlink() for s in manifest["shards"])
    with pytest.raises(ValueError, match="different data/settings"):
        script.create_subset(source, dest, seed=43)
    (dest / "samples.idx").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="index changed"):
        script.create_subset(source, dest)


def test_depth_probes_share_prefix_and_preserve_only_selected_rollout(monkeypatch):
    torch.manual_seed(42)
    torch.set_num_threads(1)
    cfg = copy.deepcopy(config(1))
    cfg.num_hidden_layers = 4
    cfg.layer_types = ["full_attention"] * 4
    target = Qwen3ForCausalLM(cfg).eval()
    draft = Qwen3DSparkModel(config()).eval()
    prompt = torch.tensor([[2, 7, 9, 3]])
    positions = torch.arange(40)[None]
    output_ids = torch.zeros_like(positions)
    output_ids[:, :4] = prompt
    target_cache, draft_cache = DynamicCache(), ContextKVCache()
    with torch.inference_mode():
        initial = target(prompt, past_key_values=target_cache, use_cache=True, output_hidden_states=True)
        output_ids[:, 4] = initial.logits[:, -1].argmax(-1)
        hidden = extract_context_feature(initial.hidden_states, draft.target_layer_ids)
    import deepspec.eval.dspark.depth_probe as probe_module
    real_verify = probe_module.verify_draft_tokens
    trial_prefix_lengths = []
    def observe_verify(**kwargs):
        trial_prefix_lengths.append(kwargs["past_key_values_target"].get_seq_length())
        return real_verify(**kwargs)
    monkeypatch.setattr(probe_module, "verify_draft_tokens", observe_verify)
    start = 4
    for _ in range(3):
        base_cache_lengths = [k.shape[-2] for k, _ in draft_cache.layers.values()]
        rows, selected = probe_depths(target=target, draft=draft, target_cache=target_cache,
            draft_cache=draft_cache, target_hidden_states=hidden, output_ids=output_ids,
            position_ids=positions, start=start, repeats=2)
        assert trial_prefix_lengths[-6:] == [start] * 6
        assert [k.shape[-2] for k, _ in draft_cache.layers.values()] == base_cache_lengths
        assert [row["num_loops"] for row in rows] == [1, 2, 3]
        committed = selected["committed_tokens"]
        expected = output_ids[:, :start + 1].clone()
        with torch.inference_mode():
            for _ in range(committed.shape[1]):
                token = target(expected).logits[:, -1].argmax(-1, keepdim=True)
                expected = torch.cat((expected, token), dim=1)
        torch.testing.assert_close(committed, expected[:, start + 1:])
        output_ids[:, start + 1:start + 1 + committed.shape[1]] = committed
        assert all(k.shape[-2] == start for k, _ in selected["draft_cache"].layers.values())
        start += committed.shape[1]
        assert target_cache.get_seq_length() == start
        hidden, draft_cache = selected["target_hidden_states"], selected["draft_cache"]
    assert not hasattr(draft, "eval_num_loops")


def test_aggregate_oracle_matches_exhaustive_search_and_ignores_terminal_rounds():
    script = load_script("summarize_prefix")
    rounds = []
    for times, progress in [([10, 13, 16], [4, 5, 7]), ([9, 14, 19], [4, 4, 4]), ([11, 12, 17], [2, 5, 6])]:
        exits = [dict(num_loops=i + 1, round_wall_ms=t, progress_tokens=p,
            accepted_draft_tokens=p - 1, progress_per_second_proxy=p * 1000 / t)
            for i, (t, p) in enumerate(zip(times, progress))]
        rounds.append(dict(exits=exits, valid_for_utility=True,
            best_loop_by_proxy=max(exits, key=lambda e: e["progress_per_second_proxy"])["num_loops"]))
    choices, rate = script.aggregate_oracle(rounds)
    exhaustive = max(sum(rounds[i]["exits"][j]["progress_tokens"] for i, j in enumerate(choice)) * 1000 /
        sum(rounds[i]["exits"][j]["round_wall_ms"] for i, j in enumerate(choice))
        for choice in itertools.product(range(3), repeat=3))
    assert rate == pytest.approx(exhaustive)
    summary = script.summarize(rounds + [dict(valid_for_utility=False)])
    assert summary["utility_rounds"] == 3 and summary["excluded_terminal_or_budget_rounds"] == 1
    assert summary["oracle_proxy_ratio_vs_best_fixed"] >= 1


def test_same_prefix_scanner_uses_real_ar_prefixes_and_excludes_truncated_rounds(tmp_path):
    torch.manual_seed(13)
    torch.set_num_threads(1)
    cfg = copy.deepcopy(config(1)); cfg.num_hidden_layers = 4
    cfg.layer_types = ["full_attention"] * 4
    target, draft = Qwen3ForCausalLM(cfg).eval(), Qwen3DSparkModel(config()).eval()
    args = SimpleNamespace(max_new_tokens=9, max_rounds=20, repeats=1)
    path = tmp_path / "trace.jsonl"
    with path.open("w") as handle:
        count = load_script("same_prefix").scan_sample(target, draft, torch.tensor([[2, 7, 9, 3]]),
            None, args, "tiny", 0, handle)
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert count == len(rows) > 0
    assert len({r["prefix_sha256"] for r in rows}) == len(rows)
    assert all(r["rollout_num_loops"] == 3 and len(r["exits"]) == 3 for r in rows)

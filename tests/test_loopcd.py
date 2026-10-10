import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from transformers import Qwen3ForCausalLM

from deepspec.eval.dspark.draft_ops import build_dspark_proposal, forward_dspark_draft_block
from deepspec.eval.dspark.evaluator import Qwen3DSparkEvaluator
from deepspec.eval.dspark.loopcd import LoopCDConfig, loopcd_logits
from deepspec.modeling.dspark.context_cache import ContextKVCache
from deepspec.modeling.dspark.qwen3.modeling import Qwen3DSparkModel
from test_loop_qwen3 import config


@pytest.fixture(autouse=True)
def deterministic():
    torch.manual_seed(42)
    torch.set_num_threads(1)


def tiny_draft():
    cfg = config(5)
    cfg.num_hidden_layers = 1
    cfg.layer_types = ["full_attention"]
    cfg.block_size = 7
    return Qwen3DSparkModel(cfg).eval()


def test_log_probability_equivalence_and_plausibility_mask():
    last = torch.tensor([[[3., 2.9, -2.]]])
    early = torch.tensor([[[5., 1., -30.]]])
    cfg = LoopCDConfig()
    scores = loopcd_logits(last, early, cfg)
    expected = last.log_softmax(-1) - cfg.strength * early.log_softmax(-1)
    expected[..., 2] = -torch.inf
    torch.testing.assert_close(scores.softmax(-1), expected.softmax(-1))
    # Contrast changes A -> B; the implausible but high-contrast C stays masked.
    assert last.argmax(-1).item() == 0 and scores.argmax(-1).item() == 1
    assert torch.isneginf(scores[..., 2]).all()
    shifted = loopcd_logits(last + 20, early - 10, cfg)
    torch.testing.assert_close(scores.softmax(-1), shifted.softmax(-1))


@pytest.mark.parametrize("kwargs", [dict(early_loop=0), dict(strength=-0.1),
    dict(strength=float("nan")), dict(alpha=0), dict(alpha=1.1), dict(alpha=float("nan"))])
def test_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        LoopCDConfig(**kwargs)


@pytest.mark.parametrize("temperature", [0.0, 0.7])
def test_proposal_uses_actual_modified_distribution_and_two_head_calls(temperature):
    draft = tiny_draft()
    draft.compute_logits = Mock(wraps=draft.compute_logits)
    early, last = torch.randn(1, 7, 32), torch.randn(1, 7, 32)
    ids = torch.full((1, 7), 63); ids[:, 0] = 2
    with torch.inference_mode():
        proposal = build_dspark_proposal(draft, draft_input_ids=ids, block_hidden=last,
            block_size=7, temperature=temperature, confidence_threshold=0,
            loopcd=LoopCDConfig(), early_block_hidden=early)
        assert draft.compute_logits.call_count == 2
        logits = loopcd_logits(draft.lm_head(last), draft.lm_head(early), LoopCDConfig())
        if temperature == 0:
            expected = torch.nn.functional.one_hot(logits.argmax(-1), 64).float()
        else:
            expected = (logits / temperature).softmax(-1)
        torch.testing.assert_close(proposal.draft_probs, expected)
        assert proposal.verify_input_ids.shape == (1, 8)
        assert proposal.verify_input_ids[0, 0] == 2
        selected = proposal.draft_probs.gather(-1, proposal.verify_input_ids[:, 1:, None])
        assert (selected > 0).all()
        baseline = build_dspark_proposal(draft, draft_input_ids=ids, block_hidden=last,
            block_size=7, temperature=temperature, confidence_threshold=0)
        assert draft.compute_logits.call_count == 3  # Disabled path has one head.
        if temperature == 0:
            control = build_dspark_proposal(draft, draft_input_ids=ids, block_hidden=last,
                block_size=7, temperature=0, confidence_threshold=0,
                loopcd=LoopCDConfig(strength=0), early_block_hidden=early)
            assert torch.equal(control.verify_input_ids, baseline.verify_input_ids)


@pytest.mark.parametrize("boundary_norm", [False, True])
def test_exit_collection_preserves_recurrence_and_projects_context_once(boundary_norm):
    draft = tiny_draft(); draft.loop_boundary_norm = boundary_norm
    attention = draft.layers[0].self_attn
    attention.project_context = Mock(wraps=attention.project_context)
    ids = torch.full((1, 7), 63); ids[:, 0] = 2
    cache = ContextKVCache()
    kwargs = dict(draft_input_ids=ids, position_ids=torch.arange(30)[None],
                  target_hidden_states=torch.randn(1, 6, 64), start=6, block_size=7)
    with torch.inference_mode():
        exits = forward_dspark_draft_block(draft, past_key_values_draft=cache,
                                           return_all_loop_hidden=True, **kwargs)
        assert len(exits) == 5 and attention.project_context.call_count == 1
        assert len(cache.layers) == 1 and cache.get_seq_length() == 6
        last = forward_dspark_draft_block(draft, past_key_values_draft=ContextKVCache(), **kwargs)
        torch.testing.assert_close(exits[-1], last)


@pytest.mark.parametrize("strength", [0.0, 0.2])
@pytest.mark.parametrize("boundary_norm", [False, True])
def test_loopcd_evaluator_greedy_output_matches_target_ar(strength, boundary_norm):
    target_cfg = copy.deepcopy(config(1))
    target_cfg.num_hidden_layers = 4
    target_cfg.layer_types = ["full_attention"] * 4
    target = Qwen3ForCausalLM(target_cfg).eval()
    draft = tiny_draft(); draft.loop_boundary_norm = boundary_norm
    evaluator = Qwen3DSparkEvaluator.__new__(Qwen3DSparkEvaluator)
    evaluator.target_model, evaluator.draft_model = target, draft
    evaluator.loopcd = LoopCDConfig(strength=strength)
    evaluator.confidence_head_recorder = None
    evaluator.args = SimpleNamespace(max_new_tokens=18, temperature=0,
                                     confidence_threshold=0, profile=True)
    prompt = torch.tensor([[2, 7, 9, 3]])
    with torch.inference_mode():
        result = evaluator.generate_one_sample(input_ids=prompt, stop_token_ids=None)
        expected = prompt.clone()
        for _ in range(18):
            expected = torch.cat((expected, target(expected).logits[:, -1].argmax(-1, keepdim=True)), dim=1)
    assert torch.equal(result.output_ids, expected)
    assert result.draft_ms > 0


def test_reject_unsupported_heads_and_invalid_exit():
    cfg = LoopCDConfig()
    with pytest.raises(ValueError, match="early_loop"):
        cfg.validate_model(tiny_draft(), 1)
    with pytest.raises(ValueError, match="Markov/confidence"):
        cfg.validate_model(Qwen3DSparkModel(config(3, dspark=True)), 3)


def test_cli_settings_and_invalid_arguments(monkeypatch):
    import eval as entrypoint
    base = ["eval.py", "--target_name_or_path", "target", "--draft_name_or_path", "draft"]
    monkeypatch.setattr("sys.argv", base + ["--loopcd", "--num-loops", "5"])
    args = entrypoint.parse_args()
    assert args.loopcd and args.loopcd_early_loop == 1 and args.loopcd_lambda == 0.2
    for extra in (["--loopcd", "--num-loops", "1"], ["--loopcd", "--loopcd-alpha", "0"],
                  ["--loopcd", "--loopcd-lambda", "nan"], ["--loopcd", "--autoregressive"]):
        monkeypatch.setattr("sys.argv", base + extra)
        with pytest.raises(SystemExit):
            entrypoint.parse_args()


def load_sweep():
    path = Path(__file__).resolve().parents[1] / "scripts/loop/loopcd_sweep.py"
    spec = importlib.util.spec_from_file_location("loopcd_sweep", path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def test_sweep_keeps_settings_paired_and_summarizes_real_metric_fields(tmp_path):
    sweep = load_sweep()
    args = sweep.parse_args(["--target", "target", "--draft", "draft",
        "--output-dir", str(tmp_path), "--repeats", "2", "--lambdas", "0.2"])
    runs = sweep.build_runs(args)
    assert len(runs) == 4
    assert runs == sweep.build_runs(args)
    assert len({run["output"] for run in runs}) == 4
    for run in runs:
        command = run["command"]
        assert command[command.index("--seed") + 1] == "42"
        assert command[command.index("--num-loops") + 1] == "5"
        assert ("--loopcd" in command) == (run["variant"] != "baseline")
        metric = dict(dataset="gsm8k", num_samples=16, acceptance_length=3,
            accept_rates_by_position=[0.8, 0.75, 1.0], conditional_accept_rates_by_position=[0.8, 0.625, 0.4],
            serial_decode_tokens_per_second=100, serial_generation_tokens_per_second=80,
            draft_ms=40, round_wall_ms=10, decode_wall_ms=100)
        if run["variant"] != "baseline":
            metric.update(acceptance_length=3.5, serial_decode_tokens_per_second=110)
        path = Path(run["output"]); path.parent.mkdir(parents=True)
        path.write_text(json.dumps(dict(metrics=[metric])), encoding="utf-8")
    summary = sweep.summarize(runs, tmp_path)
    improved = next(row for row in summary if row["variant"] != "baseline")
    assert improved["decode_tps_gain_pct"] == pytest.approx(10)
    assert improved["accept_length_delta"] == 0.5
    assert improved["accepted_draft_per_round"] == 1.5
    assert improved["draft_ms_per_round"] == 4
    assert (tmp_path / "comparison.csv").is_file() and (tmp_path / "summary.csv").is_file()

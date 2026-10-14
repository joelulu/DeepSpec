"""Actual launch options, DSpark objectives and provisional-prefix semantics."""
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from deepspec.modeling.dspark.qwen3.modeling import Qwen3DSparkModel
from deepspec.trainer.dspark_trainer import Qwen3DSparkTrainer
from deepspec.utils.config import load_config, parse_opts_to_config
from scripts.side.evaluate import build_evaluations
from test_side_loop import side_config, data
from test_loop_qwen3 import dense_mask, config


@pytest.mark.parametrize("algorithm", ["dflash", "dspark"])
@pytest.mark.parametrize("variant", ["baseline", "A", "B"])
def test_real_shell_defaults_generate_eight_gpu_three_epoch_launch(tmp_path, algorithm, variant):
    from deepspec.modeling.dspark.qwen3.config import build_draft_config
    source = tmp_path / "cache"; source.mkdir()
    (source / "manifest.json").write_text(json.dumps(dict(num_samples=10000,
        target_model_name_or_path="target", target_layer_ids=[0, 2])))
    env = dict(os.environ)
    for key in ("CUDA_VISIBLE_DEVICES", "DATA_PERCENT", "EPOCHS", "LOCAL_BATCH", "GLOBAL_BATCH",
                "BLOCK_SIZE", "SIDE_LOOPS", "GROUP_SIZE", "SMOKE_STEPS", "INIT_CHECKPOINT"):
        env.pop(key, None)
    env.update(SOURCE_CACHE=str(source), RUN_ROOT=str(tmp_path / "runs"),
               DRY_RUN="1", PYTHON=sys.executable)
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(["bash", f"train_{algorithm}_side.sh", variant], cwd=root,
        env=env, text=True, capture_output=True, check=True)
    plan, _ = json.JSONDecoder().raw_decode(result.stdout)
    assert plan["algorithm"] == algorithm and plan["variant"] == variant
    assert plan["visible_gpus"] == "0,1,2,3,4,5,6,7"
    assert plan["data_percent"] == "3" and plan["epochs"] == 3
    assert plan["predicted_tokens"] == 15 and plan["block_size"] == 16
    assert plan["steps_per_epoch"] == 4 and plan["max_train_steps"] == 12
    from scripts.side.train import build_side_launch
    env.update(DATA_PERCENT="3", EPOCHS="3", LOCAL_BATCH="2", GLOBAL_BATCH="64",
               SEED="42", HOLDOUT_SAMPLES="32", ANCHORS="2", LR="0.0006",
               SHARDING="no_shard", LOG_EVERY="5", DIAGNOSTIC_EVERY="0")
    _, command, opts = build_side_launch(variant, env, algorithm)
    cfg = parse_opts_to_config([k + "=" + json.dumps(v) for k, v in opts.items()],
                              load_config(root / command[command.index("--config") + 1]))
    draft_cfg = build_draft_config(config(1), cfg.model)
    draft = Qwen3DSparkModel(draft_cfg)
    assert (draft.markov_head is not None) == (algorithm == "dspark")
    assert (draft.confidence_head is not None) == (algorithm == "dspark")
    assert draft_cfg.draft_algorithm == algorithm and not draft.sample_from_anchor


@pytest.mark.parametrize("mode", ["none", "fusion", "grouped"])
def test_dspark_real_training_loss_has_l1_confidence_and_group_ce(monkeypatch, mode):
    import torch.distributed as dist
    import deepspec.modeling.dspark.qwen3.modeling as modeling
    torch.manual_seed(42); torch.set_num_threads(1)
    monkeypatch.setattr(modeling, "create_dspark_attention_mask", dense_mask)
    monkeypatch.setattr(dist, "get_world_size", lambda: 1)
    cfg = side_config(mode, steps=2, block=7, dspark=True)
    cfg.sample_from_anchor = False
    model = Qwen3DSparkModel(cfg).train()
    model.set_embedding_head_trainable(False)
    args = load_config("config/side/qwen3_4b.py")
    args.model.ce_loss_alpha = 0.1
    args.model.l1_loss_alpha = 0.9
    args.model.confidence_head_alpha = 1.0
    trainer = SimpleNamespace(draft_model=model, model=model, args=args,
        global_step=0, next_micro_step=0, global_rank=0, world_size=1,
        gradient_accumulation_steps=1)
    captured = []
    model.register_forward_hook(lambda module, args, output: captured.append(output))
    batch = dict(input_ids=torch.randint(0, 60, (1, 14)), loss_mask=torch.ones(1, 14),
                 target_hidden_states=torch.randn(1, 14, 64),
                 target_last_hidden_states=torch.randn(1, 14, 32))
    loss = Qwen3DSparkTrainer.run_batch(trainer, batch)
    output = captured[0]
    weights = output.eval_mask.float() * torch.exp(-torch.arange(7).float() / 4)
    def weighted(value):
        return (value * weights).sum() / (weights.sum() + 1e-6)
    ce = F.cross_entropy(output.draft_logits.flatten(0, 2), output.target_ids.flatten(),
                         reduction="none").reshape_as(weights)
    probability_l1 = (output.draft_logits.softmax(-1) - output.aligned_target_logits.softmax(-1)).abs().sum(-1)
    confidence_targets = (1 - 0.5 * probability_l1).clamp(0, 1).detach()
    confidence = F.binary_cross_entropy_with_logits(output.confidence_pred,
        confidence_targets, reduction="none")
    expected = 0.1 * weighted(ce) + 0.9 * weighted(probability_l1) + weighted(confidence)
    if mode == "grouped":
        auxiliary = F.cross_entropy(output.side_aux_logits.flatten(0, 2), output.target_ids.flatten(),
                                    reduction="none").reshape_as(weights)
        expected = expected + 0.2 * weighted(auxiliary)
    torch.testing.assert_close(loss, expected)
    assert not output.eval_mask[:, :, 0].any()
    loss.backward()
    assert model.markov_head.markov_w2.weight.grad.abs().sum() > 0
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.confidence_head.parameters())
    assert model.layers[0].self_attn.q_proj.weight.grad.abs().sum() > 0
    assert model.lm_head.weight.grad is None


def test_grouped_dspark_markov_uses_branch_predictions_and_preserves_blocks(monkeypatch):
    torch.set_num_threads(1)
    cfg = side_config("grouped", steps=2, block=7, dspark=True)
    cfg.sample_from_anchor = False
    model = Qwen3DSparkModel(cfg).eval()
    previous, inputs = [], []
    def forced_step(logits, *, token_ids, hidden_states):
        index = len(previous)
        previous.append(token_ids.clone())
        ids = torch.tensor([7 + index, 20 + index])
        return logits.new_zeros(logits.shape).scatter(-1, ids.unsqueeze(-1), 100)
    monkeypatch.setattr(model.markov_head, "apply_step_logits", forced_step)
    model.layers[0].register_forward_pre_hook(lambda module, args, kwargs:
        inputs.append(kwargs["hidden_states"].clone()), with_kwargs=True)
    with torch.no_grad():
        model._forward_backbone(**data(model, blocks=2))
    assert [value.tolist() for value in previous] == [[2, 2], [7, 20], [8, 21], [9, 22], [10, 23], [11, 24]]
    torch.testing.assert_close(inputs[1].reshape(1, 2, 7, 32)[:, :, 1:4],
        model.embed_tokens(torch.tensor([[[7, 8, 9], [20, 21, 22]]])))


@pytest.mark.parametrize("algorithm", ["dflash", "dspark"])
@pytest.mark.parametrize("variant", ["baseline", "A", "B"])
def test_evaluation_selects_algorithm_schedules_and_confidence_threshold(tmp_path, algorithm, variant):
    cfg = dict(num_hidden_layers=5, num_loops=1, block_size=16,
               markov_rank=256 if algorithm == "dspark" else 0,
               enable_confidence_head=algorithm == "dspark",
               side_loop_mode={"baseline": "none", "A": "fusion", "B": "grouped"}[variant])
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    env = dict(TARGET_MODEL="target", RUN_ROOT=str(tmp_path / "results"), EVAL_REPEATS="1",
               CONFIDENCE_THRESHOLD="0.5" if algorithm == "dspark" else "0")
    _, commands = build_evaluations(tmp_path, env, algorithm, "both")
    assert len(commands) == (1 if variant == "baseline" else 2)
    assert commands[0][-1] == "serial"
    if variant != "baseline":
        assert commands[1][-1] == "overlap"
    assert float(commands[0][commands[0].index("--confidence-threshold") + 1]) == float(env["CONFIDENCE_THRESHOLD"])
    with pytest.raises(ValueError, match="checkpoint is"):
        build_evaluations(tmp_path, env, "dspark" if algorithm == "dflash" else "dflash")

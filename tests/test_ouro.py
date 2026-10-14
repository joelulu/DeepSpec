import copy
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import os
import subprocess

import pytest
import torch
from transformers import Qwen3ForCausalLM

from deepspec.modeling.dspark.ouro import (block_cross_entropy, compute_ouro_loss,
                                         exit_distribution)
from deepspec.modeling.dspark.qwen3.modeling import Qwen3DSparkModel
from deepspec.modeling.dspark.context_cache import ContextKVCache
from deepspec.eval.dspark.evaluator import Qwen3DSparkEvaluator
from deepspec.trainer.dspark_trainer import Qwen3DSparkTrainer
from deepspec.trainer.base_trainer import BaseTrainer
from deepspec.utils.config import ConfigNode
from test_loop_qwen3 import config, inputs, dense_mask
from test_loop_pilot import load_script


@pytest.fixture(autouse=True)
def deterministic():
    from deepspec.utils.metrics import reset
    torch.manual_seed(42)
    torch.set_num_threads(1)
    reset()
    yield
    reset()


def gate_config(mode="joint", boundary=False):
    cfg = config(5)
    cfg.num_hidden_layers = 1
    cfg.layer_types = ["full_attention"]
    cfg.loop_loss_weights = [0, 0, 0, 0, 1]
    cfg.enable_exit_gate = True
    cfg.min_exit_loop = 3
    cfg.ouro_training_mode = mode
    cfg.ouro_beta = 0.1
    cfg.gate_gain_slope = 50.0
    cfg.gate_gain_threshold = 0.005
    cfg.loop_boundary_norm = boundary
    cfg.exit_gate_trained = True
    return cfg


def batch():
    return dict(input_ids=torch.randint(0, 60, (2, 12)), loss_mask=torch.ones(2, 12),
                target_hidden_states=torch.randn(2, 12, 64))


def test_pdf_uses_hazards_and_assigns_remainder_to_final_exit():
    logits = torch.logit(torch.tensor([[0.2, 0.5], [0.99, 0.01]])).requires_grad_()
    pdf = exit_distribution(logits)
    torch.testing.assert_close(pdf[0], torch.tensor([0.2, 0.4, 0.4]))
    torch.testing.assert_close(pdf.sum(-1), torch.ones(2))
    (pdf * torch.tensor([3., 2., 1.])).sum().backward()
    assert torch.isfinite(logits.grad).all() and logits.grad.abs().sum() > 0


@pytest.mark.parametrize("boundary", [False, True])
@pytest.mark.parametrize("q,depth", [(0.5, 3), (0.7, 4), (0.95, 5)])
def test_q_exit_matches_fixed_depth_and_skips_actual_layer_calls(boundary, q, depth):
    model = Qwen3DSparkModel(gate_config(boundary=boundary)).eval()
    with torch.no_grad():
        model.exit_gate.weight.zero_(); model.exit_gate.bias.zero_()
    calls = []
    hook = model.layers[0].register_forward_hook(lambda *args: calls.append(1))
    data = inputs()
    with torch.no_grad():
        dynamic = model._forward_backbone(**data, exit_q=q)
        assert len(calls) == depth and model.last_exit_depth == depth
        calls.clear()
        fixed = model._forward_backbone(**data, num_loops=depth)
    torch.testing.assert_close(dynamic, fixed)
    hook.remove()


@pytest.mark.parametrize("boundary", [False, True])
def test_gate_disabled_matches_baseline_and_old_checkpoint_has_no_gate(boundary):
    cfg = gate_config(boundary=boundary)
    gated = Qwen3DSparkModel(cfg).eval()
    plain_cfg = copy.deepcopy(cfg); plain_cfg.enable_exit_gate = False; plain_cfg.ouro_training_mode = "baseline"
    baseline = Qwen3DSparkModel(plain_cfg).eval()
    baseline.load_state_dict({k: v for k, v in gated.state_dict().items() if not k.startswith("exit_gate.")})
    data = inputs()
    with torch.no_grad():
        torch.testing.assert_close(gated._forward_backbone(**data), baseline._forward_backbone(**data))
    assert baseline.exit_gate is None
    assert not any(k.startswith("exit_gate.") for k in baseline.state_dict())


def test_joint_updates_gate_and_backbone_even_with_zero_legacy_exit_weights(monkeypatch):
    import deepspec.modeling.dspark.qwen3.modeling as modeling
    monkeypatch.setattr(modeling, "create_dspark_attention_mask", dense_mask)
    model = Qwen3DSparkModel(gate_config())
    model.set_embedding_head_trainable(False)
    outputs = model(**batch())
    for output in outputs[2:]:
        output.draft_logits.retain_grad()
    loss = compute_ouro_loss(outputs, model.config)
    loss.backward()
    assert torch.isfinite(loss)
    assert model.exit_gate.weight.grad.abs().sum() > 0
    assert model.layers[0].self_attn.q_proj.weight.grad.abs().sum() > 0
    assert model.fc.weight.grad.abs().sum() > 0
    assert model.embed_tokens.weight.grad is None and model.lm_head.weight.grad is None
    assert all(o.draft_logits.grad.abs().sum() > 0 for o in outputs[2:])


def test_frozen_training_only_changes_gate_and_retains_initial_gate_for_b2(monkeypatch):
    import deepspec.modeling.dspark.qwen3.modeling as modeling
    monkeypatch.setattr(modeling, "create_dspark_attention_mask", dense_mask)
    model = Qwen3DSparkModel(gate_config("gate_only"))
    trainer = Qwen3DSparkTrainer.__new__(Qwen3DSparkTrainer)
    trainer.args = SimpleNamespace(model=dict(ouro_training_mode="gate_only", init_checkpoint="source", l1_loss_alpha=0))
    trainer.draft_model = model
    gate_before = model.exit_gate.weight.detach().clone()
    trainer.prepare_model_for_training()
    torch.testing.assert_close(model.exit_gate.weight, gate_before)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.01)
    outputs = model(**batch())
    assert not any(o.draft_logits.requires_grad for o in outputs)
    loss = compute_ouro_loss(outputs, model.config)
    loss.backward(); optimizer.step()
    changed = {k for k, v in model.state_dict().items() if not torch.equal(v, before[k])}
    assert changed and changed <= {"exit_gate.weight", "exit_gate.bias"}


def test_block_ce_ignores_invalid_slots_and_dummy_blocks():
    logits = torch.randn(1, 2, 3, 8)
    out = SimpleNamespace(draft_logits=logits, target_ids=torch.zeros(1, 2, 3, dtype=torch.long),
        eval_mask=torch.tensor([[[1, 0, 0], [0, 0, 0]]]), block_keep_mask=torch.tensor([[True, False]]))
    ce, valid = block_cross_entropy(out)
    torch.testing.assert_close(ce[0, 0], torch.nn.functional.cross_entropy(logits[0, 0, :1], out.target_ids[0, 0, :1]))
    assert valid.tolist() == [[True, False]]
    assert ce[0, 1] == 0


@pytest.mark.parametrize("source_has_gate", [False, True])
def test_initialization_loads_real_checkpoint_and_preserves_existing_weights(monkeypatch, tmp_path, source_has_gate):
    import deepspec.trainer.base_trainer as base
    cfg = gate_config("joint")
    cfg.enable_exit_gate = source_has_gate
    if not source_has_gate:
        cfg.ouro_training_mode = "baseline"
    source = Qwen3DSparkModel(cfg)
    source.save_pretrained(tmp_path)
    desired = gate_config("gate_only")
    desired.exit_gate_trained = False
    real_auto_config = base.AutoConfig.from_pretrained
    monkeypatch.setattr(base.AutoConfig, "from_pretrained", lambda path: config(1) if path == "target" else real_auto_config(path))
    monkeypatch.setattr(base.AutoTokenizer, "from_pretrained", lambda path: "tokenizer")
    trainer = BaseTrainer.__new__(BaseTrainer)
    trainer.device = torch.device("cpu"); trainer.precision_dtype = torch.float32
    trainer.args = SimpleNamespace(model=ConfigNode(target_model_name_or_path="target", init_checkpoint=str(tmp_path)))
    trainer._build_draft_model = lambda **kwargs: Qwen3DSparkModel(desired)
    loaded, tokenizer = trainer.build_models()
    assert tokenizer == "tokenizer" and loaded.config.ouro_training_mode == "gate_only"
    assert loaded.exit_gate is not None
    for name, value in source.state_dict().items():
        torch.testing.assert_close(value, loaded.state_dict()[name])
    assert loaded.embed_tokens.weight.requires_grad is False
    assert loaded.lm_head.weight.requires_grad is False


def test_gate_label_uses_current_to_next_gain_and_not_final_loss_only():
    cfg = gate_config("gate_only")
    ce_values = [5., 4., 3., 2., 2.]
    outputs = []
    for value in ce_values:
        # Two-class logits giving exactly the requested CE for class zero.
        logits = torch.tensor([0., torch.log(torch.expm1(torch.tensor(value)))])[None, None, None, :]
        outputs.append(SimpleNamespace(draft_logits=logits, target_ids=torch.zeros(1, 1, 1, dtype=torch.long),
            eval_mask=torch.ones(1, 1, 1), block_keep_mask=torch.ones(1, 1, dtype=torch.bool),
            exit_gate_logits=torch.zeros(1, 1, requires_grad=True)))
    loss = compute_ouro_loss(outputs, cfg)
    loss.backward()
    assert outputs[2].exit_gate_logits.grad.item() > 0  # reduce stop probability at loop3
    assert outputs[3].exit_gate_logits.grad.item() < 0  # increase stop probability at loop4


def test_joint_objective_is_weighted_losses_not_ce_of_mixed_logits():
    cfg = gate_config("joint")
    logits = [torch.tensor([0., 0.]), torch.tensor([0., 0.]), torch.tensor([4., -1.]),
              torch.tensor([-1., 4.]), torch.tensor([1., 0.])]
    outputs = [SimpleNamespace(draft_logits=z[None, None, None, :],
        target_ids=torch.zeros(1, 1, 1, dtype=torch.long), eval_mask=torch.ones(1, 1, 1),
        block_keep_mask=torch.ones(1, 1, dtype=torch.bool), exit_gate_logits=torch.zeros(1, 1)) for z in logits]
    loss = compute_ouro_loss(outputs, cfg)
    pdf = torch.tensor([.5, .25, .25])
    ces = torch.stack([torch.nn.functional.cross_entropy(z[None], torch.tensor([0])) for z in logits[2:]])
    expected = (pdf * ces).sum() + cfg.ouro_beta * (pdf * pdf.log()).sum()
    torch.testing.assert_close(loss, expected)


def test_save_reload_preserves_dynamic_exit_and_cache_across_blocks(tmp_path):
    model = Qwen3DSparkModel(gate_config()).eval()
    with torch.no_grad():
        model.exit_gate.weight.zero_(); model.exit_gate.bias.fill_(10)
    model.save_pretrained(tmp_path)
    restored = Qwen3DSparkModel.from_pretrained(tmp_path, attn_implementation="sdpa").eval()
    cache = ContextKVCache()
    data = inputs()
    with torch.no_grad():
        first = restored._forward_backbone(**data, exit_q=.8, past_key_values=cache)
        torch.testing.assert_close(first, model._forward_backbone(**data, num_loops=3))
        context = torch.randn(1, 2, 64)
        noise = torch.randn(1, 3, 32)
        actual = restored._forward_backbone(target_hidden_states=context, noise_embedding=noise,
            position_ids=torch.arange(6, 11)[None], past_key_values=cache, exit_q=.8)
        expected = model._forward_backbone(target_hidden_states=torch.cat((data["target_hidden_states"], context), 1),
            noise_embedding=noise, position_ids=torch.arange(11)[None], num_loops=3)
    torch.testing.assert_close(actual, expected, atol=3e-6, rtol=3e-5)
    assert restored.last_exit_depth == 3
    assert all(k.shape[-2] == 8 for k, _ in cache.layers.values())


@pytest.mark.parametrize("bias", [-10., 10.])
def test_gated_speculative_generation_matches_target_ar(bias):
    target_cfg = copy.deepcopy(config(1)); target_cfg.num_hidden_layers = 4
    target_cfg.layer_types = ["full_attention"] * 4
    target = Qwen3ForCausalLM(target_cfg).eval()
    draft = Qwen3DSparkModel(gate_config()).eval(); draft.eval_exit_q = .8
    with torch.no_grad():
        draft.exit_gate.weight.zero_(); draft.exit_gate.bias.fill_(bias)
    evaluator = Qwen3DSparkEvaluator.__new__(Qwen3DSparkEvaluator)
    evaluator.target_model, evaluator.draft_model = target, draft
    evaluator.loopcd = None; evaluator.confidence_head_recorder = None
    evaluator.exit_depth_hist = torch.zeros(5, dtype=torch.long)
    evaluator.args = SimpleNamespace(max_new_tokens=12, temperature=0, confidence_threshold=0, profile=True)
    prompt = torch.tensor([[2, 7, 9, 3]])
    with torch.no_grad():
        result = evaluator.generate_one_sample(input_ids=prompt, stop_token_ids=None)
        expected = prompt.clone()
        for _ in range(12):
            expected = torch.cat((expected, target(expected).logits[:, -1].argmax(-1, keepdim=True)), 1)
    assert torch.equal(result.output_ids, expected)
    assert evaluator.exit_depth_hist.sum() > 0
    assert evaluator.exit_depth_hist[2 if bias > 0 else 4] > 0


def test_inference_rejects_untrained_gate_and_training_early_exit():
    model = Qwen3DSparkModel(gate_config())
    with pytest.raises(ValueError, match="inference-only"):
        model._forward_backbone(**inputs(), exit_q=.8)
    model.eval(); model.config.exit_gate_trained = False
    with pytest.raises(ValueError, match="trained exit gate"):
        model._forward_backbone(**inputs(), exit_q=.8)


def launch_environment(tmp_path):
    source = tmp_path / "cache"; source.mkdir()
    (source / "manifest.json").write_text(json.dumps(dict(num_samples=1000, target_layer_ids=[0, 2], target_model_name_or_path="target")))
    return dict(SOURCE_CACHE=str(source), DATA_PERCENT="10", SEED="42", HOLDOUT_SAMPLES="4",
        EPOCHS="10", LOCAL_BATCH="2", GLOBAL_BATCH="16", LOOP_TRANSFER="raw", LOOP_LOSS="late",
        LOG_EVERY="5", DIAGNOSTIC_EVERY="20", RUN_ROOT=str(tmp_path / "runs"), ANCHORS="2", LR="6e-4",
        SHARDING="no_shard", GATE_EPOCHS="1", GATE_LR="1e-3", MIN_EXIT_LOOP="3")


def test_launch_stages_infer_source_config_and_keep_separate_schedules(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts" / "loop"))
    launcher = load_script("launch_ouro")
    env = launch_environment(tmp_path)
    baseline, _, plain = launcher.build_stage("baseline", 1, 5, env)
    joint, _, trained = launcher.build_stage("joint", 1, 5, env)
    cfg = dict(num_hidden_layers=1, num_loops=5, loop_boundary_norm=True, block_size=3,
               mask_token_id=63, target_layer_ids=[0, 2], loop_loss_weights=[0, 0, 0, 0, 1])
    frozen, _, gate = launcher.build_stage("gate_only", None, None, env, "source", cfg)
    assert len({p["checkpoint_root"] for p in (baseline, joint, frozen)}) == 3
    assert plain["model.enable_exit_gate"] is False and trained["model.enable_exit_gate"] is True
    assert gate["model.loop_boundary_norm"] is True and gate["model.block_size"] == 3
    assert gate["train.num_train_epochs"] == 1 and gate["train.lr"] == .001
    assert trained["train.num_train_epochs"] == 10 and trained["train.lr"] == .0006


def test_eval_plans_force_fixed_mode_and_sweep_gates(tmp_path):
    script = load_script("eval_ouro")
    checkpoint = tmp_path / "model"; checkpoint.mkdir()
    (checkpoint / "config.json").write_text(json.dumps(dict(num_loops=5, enable_exit_gate=True, exit_gate_trained=True)))
    env = dict(TARGET_MODEL="target", RUN_ROOT=str(tmp_path), EVAL_REPEATS="2", EXIT_Q_LIST="0.5 0.8")
    root, commands = script.build_evaluations("both", str(checkpoint), env)
    assert len(commands) == 14
    assert all("--exit-gate" not in c for c in commands[:5])
    assert all("--exit-gate" in c for c in commands[5:7])
    assert root.parent.name == "results"
    (checkpoint / "config.json").write_text(json.dumps(dict(num_loops=5)))
    with pytest.raises(ValueError, match="trained exit gate"):
        script.build_evaluations("gate", str(checkpoint), env)


def test_b_stops_on_joint_failure_and_chains_checkpoint(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts" / "loop"))
    script = load_script("launch_ouro")
    monkeypatch.setattr(sys, "argv", ["launch_ouro.py", "B", "1", "5"])
    calls = []
    def failed(stage, *args, **kwargs):
        calls.append(stage)
        raise RuntimeError("joint failed")
    monkeypatch.setattr(script, "execute_stage", failed)
    with pytest.raises(RuntimeError, match="joint failed"):
        script.main()
    assert calls == ["joint"]
    calls.clear()
    def succeeded(stage, layers, loops, env, checkpoint=None, source_config=None):
        calls.append((stage, checkpoint, env.get("GATE_LABEL")))
        return tmp_path / "joint" / "step_latest", {"model.num_loops": 5}
    monkeypatch.setattr(script, "execute_stage", succeeded)
    script.main()
    assert calls[1] == ("gate_only", tmp_path / "joint" / "step_latest", "B_gate")


def test_legacy_python_launchers_dry_run_and_generated_training_options(tmp_path):
    from deepspec.utils.config import load_config, parse_opts_to_config
    from deepspec.modeling.dspark.qwen3.config import build_draft_config
    root = Path(__file__).resolve().parents[1]
    env = dict(os.environ, **launch_environment(tmp_path), DRY_RUN="1", PYTHON=sys.executable)
    result = subprocess.run([sys.executable, "scripts/loop/launch_ouro.py", "B", "1", "5"], cwd=root,
        env=env, text=True, capture_output=True, check=True)
    decoder = json.JSONDecoder()
    first, end = decoder.raw_decode(result.stdout)
    second, _ = decoder.raw_decode(result.stdout[end:].lstrip())
    assert first["plan"]["stage"] == "joint" and second["plan"]["stage"] == "gate_only"
    assert first["plan"]["checkpoint_root"] in second["overrides"]["model.init_checkpoint"]
    for launch in (first, second):
        command = launch["command"]
        opts = [command[i + 1] for i, value in enumerate(command) if value == "--opts"]
        cfg = parse_opts_to_config(opts, load_config(root / "config/loop/dflash_flex_qwen3_4b.py"))
        draft_cfg = build_draft_config(config(1), cfg.model)
        assert draft_cfg.enable_exit_gate and draft_cfg.min_exit_loop == 3
        assert cfg.logging.checkpoint_dir.startswith(launch["overrides"]["run_root"])
    checkpoint = tmp_path / "checkpoint"; checkpoint.mkdir()
    (checkpoint / "config.json").write_text(json.dumps(dict(num_loops=5, enable_exit_gate=True, exit_gate_trained=True)))
    result = subprocess.run([sys.executable, "scripts/loop/eval_ouro.py", "both", str(checkpoint)], cwd=root,
        env=dict(env, TARGET_MODEL="target", EVAL_REPEATS="1"), text=True, capture_output=True, check=True)
    plan = json.loads(result.stdout)
    assert len(plan["commands"]) == 8


def test_resume_rejects_cross_stage_and_refreezes_gate_only(tmp_path):
    from deepspec.trainer.ckpt_manager import load_resume_draft_model
    source = Qwen3DSparkModel(gate_config("gate_only"))
    source.save_pretrained(tmp_path)
    (tmp_path / "checkpoint_meta.json").write_text(json.dumps(dict(training_complete=True, save_training_state=False)))
    resumed = load_resume_draft_model(resume_checkpoint_dir=str(tmp_path), draft_model=source,
        device="cpu", precision_dtype=torch.float32, global_rank=0)
    trainer = Qwen3DSparkTrainer.__new__(Qwen3DSparkTrainer)
    trainer.args = SimpleNamespace(model=ConfigNode(ouro_training_mode="gate_only", init_checkpoint="source", l1_loss_alpha=0))
    trainer.draft_model = resumed
    trainer.prepare_model_for_training()
    assert {name for name, p in resumed.named_parameters() if p.requires_grad} == {"exit_gate.weight", "exit_gate.bias"}
    with pytest.raises(ValueError, match="ouro_training_mode"):
        load_resume_draft_model(resume_checkpoint_dir=str(tmp_path), draft_model=Qwen3DSparkModel(gate_config("joint")),
            device="cpu", precision_dtype=torch.float32, global_rank=0)

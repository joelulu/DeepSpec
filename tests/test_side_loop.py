import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from transformers import Qwen3ForCausalLM

from deepspec.modeling.dspark.qwen3.modeling import Qwen3DSparkModel
from deepspec.modeling.dspark.context_cache import ContextKVCache
from deepspec.modeling.dspark.common import extract_context_feature
from deepspec.eval.base_evaluator import generate_decoding_sample
from deepspec.eval.dspark.draft_ops import forward_dspark_draft_block, build_dspark_proposal
from test_loop_qwen3 import config, dense_mask


def side_config(mode="fusion", steps=5, block=15, dspark=False):
    cfg = config(1, dspark=dspark)
    cfg.block_size = block
    cfg.side_loop_mode = mode
    cfg.side_loop_steps = steps
    cfg.side_group_size = 3
    cfg.side_fusion_rank = 8
    cfg.side_aux_loss_alpha = 0.2
    return cfg


@pytest.fixture(autouse=True)
def deterministic():
    torch.manual_seed(42)
    torch.set_num_threads(1)


def data(model, ctx=8, blocks=1):
    ids = torch.full((1, blocks * model.block_size), model.mask_token_id)
    ids[:, ::model.block_size] = 2
    return dict(noise_embedding=model.embed_tokens(ids),
                side_anchor_token_ids=ids[:, ::model.block_size],
                target_hidden_states=torch.randn(1, ctx, 64),
                position_ids=torch.cat((torch.arange(ctx),
                    torch.arange(ctx, ctx + model.block_size).repeat(blocks))).unsqueeze(0))


@pytest.mark.parametrize("mode", ["fusion", "grouped"])
def test_initialization_matches_main_and_exact_layer_and_head_counts(mode):
    model = Qwen3DSparkModel(side_config(mode)).eval()
    baseline = Qwen3DSparkModel(config(1)).eval()
    baseline.block_size = model.block_size
    baseline.load_state_dict({key: value for key, value in model.state_dict().items()
                              if not key.startswith("side_fusion.")})
    batch = data(model)
    layer_calls, head_calls = [0] * 5, []
    hooks = []
    for index, layer in enumerate(model.layers):
        def count(module, args, output, index=index):
            layer_calls[index] += 1
        hooks.append(layer.register_forward_hook(count))
    hooks.append(model.lm_head.register_forward_hook(lambda *args: head_calls.append(1)))
    with torch.no_grad():
        actual = model._forward_backbone(**batch)
        expected = baseline._forward_backbone(**batch)
    torch.testing.assert_close(actual, expected, rtol=3e-5, atol=3e-6)
    assert layer_calls == [5, 1, 1, 1, 1]
    assert len(head_calls) == (5 if mode == "grouped" else 0)
    assert len(model.layers) == 5
    assert not any("side_layer" in name for name, _ in model.named_parameters())
    for hook in hooks:
        hook.remove()


def test_same_seed_preserves_identical_main_weights_across_variants():
    torch.manual_seed(42)
    baseline = Qwen3DSparkModel(config(1))
    for mode in ("fusion", "grouped"):
        torch.manual_seed(42)
        model = Qwen3DSparkModel(side_config(mode))
        for key, value in baseline.state_dict().items():
            torch.testing.assert_close(value, model.state_dict()[key], atol=0, rtol=0)


def test_grouped_feedback_uses_next_token_shift_and_does_not_cross_blocks(monkeypatch):
    model = Qwen3DSparkModel(side_config("grouped", steps=2, block=6)).eval()
    captured = []
    model.layers[0].register_forward_pre_hook(
        lambda module, args, kwargs: captured.append(kwargs["hidden_states"].detach().clone()), with_kwargs=True)
    def forced_logits(hidden):
        logits = hidden.new_zeros(*hidden.shape[:-1], 64)
        ids = torch.tensor([[[7, 8, 9], [10, 11, 12]]], device=hidden.device)
        return logits.scatter(-1, ids.unsqueeze(-1), 100)
    monkeypatch.setattr(model, "compute_logits", forced_logits)
    with torch.no_grad():
        model._forward_backbone(**data(model, blocks=2))
    second = captured[1].reshape(1, 2, 6, 32)
    torch.testing.assert_close(second[:, :, 1:4], model.embed_tokens(torch.tensor([[[7, 8, 9], [10, 11, 12]]])))
    torch.testing.assert_close(second[:, :, 0], model.embed_tokens(torch.tensor([[2, 2]])))


@pytest.mark.parametrize("mode", ["fusion", "grouped"])
@pytest.mark.parametrize("checkpointing", [False, True])
def test_training_backward_and_checkpoint_roundtrip(monkeypatch, tmp_path, mode, checkpointing):
    import deepspec.modeling.dspark.qwen3.modeling as modeling
    monkeypatch.setattr(modeling, "create_dspark_attention_mask", dense_mask)
    model = Qwen3DSparkModel(side_config(mode, steps=2, block=6)).train()
    model.set_embedding_head_trainable(False)
    if checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    output = model(input_ids=torch.randint(0, 60, (1, 14)), loss_mask=torch.ones(1, 14),
        target_hidden_states=torch.randn(1, 14, 64))
    loss = F.cross_entropy(output.draft_logits.flatten(0, 2), output.target_ids.flatten())
    if mode == "grouped":
        assert output.side_aux_logits.shape == output.draft_logits.shape
        loss = loss + 0.2 * F.cross_entropy(output.side_aux_logits.flatten(0, 2), output.target_ids.flatten())
    else:
        assert output.side_aux_logits is None
    loss.backward()
    assert model.side_fusion.up.weight.grad.abs().sum() > 0
    assert all(layer.self_attn.q_proj.weight.grad is not None for layer in model.layers)
    assert model.lm_head.weight.grad is None and model.embed_tokens.weight.grad is None
    with torch.no_grad():
        model.side_fusion.up.weight.add_(0.01)
    model.eval()
    batch = data(model)
    expected = model._forward_backbone(**batch)
    model.save_pretrained(tmp_path)
    restored = Qwen3DSparkModel.from_pretrained(tmp_path, attn_implementation="sdpa").eval()
    torch.testing.assert_close(restored._forward_backbone(**batch), expected)
    assert restored.side_loop_mode == mode and restored.side_loop_steps == 2


@pytest.mark.parametrize("mode", ["fusion", "grouped"])
def test_context_cache_reused_once_per_layer_and_future_features_cannot_leak(mode):
    model = Qwen3DSparkModel(side_config(mode, steps=2, block=6)).eval()
    with torch.no_grad():
        model.side_fusion.up.weight.fill_(0.1)
        batch = data(model)
        cache = ContextKVCache()
        model._forward_backbone(**batch, past_key_values=cache)
        assert len(cache.layers) == 5
        assert all(key.shape[-2] == 8 for key, _ in cache.layers.values())
        new_features = torch.randn(1, 2, 64)
        incremental = model._forward_backbone(noise_embedding=batch["noise_embedding"],
            target_hidden_states=new_features, position_ids=torch.arange(8, 16).unsqueeze(0),
            past_key_values=cache)
        complete = model._forward_backbone(noise_embedding=batch["noise_embedding"],
            target_hidden_states=torch.cat((batch["target_hidden_states"], new_features), dim=1),
            position_ids=torch.arange(16).unsqueeze(0))
        torch.testing.assert_close(incremental, complete, atol=3e-6, rtol=3e-5)
        assert all(key.shape[-2] == 10 for key, _ in cache.layers.values())
        batch = data(model)
        batch["attention_mask"] = dense_mask(anchor_positions=torch.tensor([[4]]),
            block_keep_mask=torch.tensor([[True]]), seq_len=8, block_size=6, device=torch.device("cpu"))
        batch["position_ids"] = torch.cat((torch.arange(8), torch.arange(4, 10))).unsqueeze(0)
        before = model._forward_backbone(**batch)
        batch["target_hidden_states"][:, 4:] = torch.randn(1, 4, 64) * 100
        after = model._forward_backbone(**batch)
        torch.testing.assert_close(before, after)


@pytest.mark.parametrize("mode", ["fusion", "grouped"])
@pytest.mark.parametrize("sample_from_anchor", [True, False])
@pytest.mark.parametrize("dspark", [False, True])
def test_speculative_greedy_output_matches_target(mode, sample_from_anchor, dspark):
    target_cfg = copy.deepcopy(config(1))
    target_cfg.num_hidden_layers = 4
    target_cfg.layer_types = ["full_attention"] * 4
    target = Qwen3ForCausalLM(target_cfg).eval()
    block = 6 + int(not sample_from_anchor)
    draft_cfg = side_config(mode, steps=2, block=block, dspark=dspark)
    draft_cfg.sample_from_anchor = sample_from_anchor
    draft = Qwen3DSparkModel(draft_cfg).eval()
    with torch.no_grad():
        draft.side_fusion.up.weight.fill_(0.03)
    prompt = torch.tensor([[2, 7, 9, 3]])
    def init_context(initial_output, **kwargs):
        return SimpleNamespace(past_key_values_draft=ContextKVCache(),
            target_hidden_states=extract_context_feature(initial_output.hidden_states, draft.target_layer_ids))
    def propose(context, output_ids, position_ids, start, stop_token_ids):
        ids = torch.full((1, block), 63); ids[:, 0] = output_ids[:, start]
        hidden = forward_dspark_draft_block(draft, draft_input_ids=ids, position_ids=position_ids,
            past_key_values_draft=context.past_key_values_draft, target_hidden_states=context.target_hidden_states,
            start=start, block_size=block)
        return build_dspark_proposal(draft, draft_input_ids=ids, block_hidden=hidden, block_size=block,
                                    temperature=0.0, confidence_threshold=0.0)
    def update(context, verification):
        context.target_hidden_states = extract_context_feature(verification.target_output.hidden_states,
            draft.target_layer_ids)[:, :verification.accepted_draft_tokens + 1]
    result = generate_decoding_sample(target_model=target, input_ids=prompt, max_new_tokens=12,
        max_proposal_tokens=6, temperature=0.0, stop_token_ids=None,
        init_context=init_context, propose=propose, update=update, profile=True)
    expected = prompt.clone()
    with torch.no_grad():
        for _ in range(12):
            expected = torch.cat((expected, target(expected).logits[:, -1].argmax(-1, keepdim=True)), dim=1)
    assert torch.equal(result.output_ids, expected)


def test_invalid_layout_and_cpu_overlap_are_rejected():
    with pytest.raises(ValueError, match="block_size"):
        Qwen3DSparkModel(side_config("grouped", block=16))
    cfg = side_config(); cfg.num_hidden_layers = 3
    with pytest.raises(ValueError, match="five main layers"):
        Qwen3DSparkModel(cfg)
    model = Qwen3DSparkModel(side_config()).eval()
    model.side_execution = "overlap"
    with torch.no_grad(), pytest.raises(ValueError, match="CUDA eval"):
        model._forward_backbone(**data(model))


def test_launch_defaults_and_equal_length_controls(tmp_path):
    from scripts.side.train import build_side_launch
    source = tmp_path / "cache"; source.mkdir()
    (source / "manifest.json").write_text(json.dumps(dict(num_samples=10000,
        target_model_name_or_path="target", target_layer_ids=[0, 2])))
    env = dict(SOURCE_CACHE=str(source), DATA_PERCENT="3", HOLDOUT_SAMPLES="32", SEED="42",
        EPOCHS="1", LOCAL_BATCH="2", GLOBAL_BATCH="64", LOOP_TRANSFER="raw", LOOP_LOSS="final",
        RUN_ROOT=str(tmp_path / "runs"), ANCHORS="2", LOG_EVERY="5", DIAGNOSTIC_EVERY="0",
        LR="0.0006", SHARDING="no_shard")
    for variant, block in (("baseline", 16), ("A", 16), ("B", 16)):
        plan, command, opts = build_side_launch(variant, env)
        assert opts["model.num_draft_layers"] == 5 and opts["model.num_loops"] == 1
        assert plan["block_size"] == block
        assert plan["predicted_tokens"] == 15 and opts["model.sample_from_anchor"] is False
        assert Path(plan["checkpoint_root"]).parent == tmp_path / "runs" / "deepspec"
    _, _, opts = build_side_launch("A", dict(env, BLOCK_SIZE="15"))
    assert opts["model.block_size"] == 15
    with pytest.raises(ValueError, match="BLOCK_SIZE"):
        build_side_launch("B", dict(env, BLOCK_SIZE="15"))


@pytest.mark.parametrize("mode", ["fusion", "grouped"])
def test_warmstart_from_plain_checkpoint_adds_only_zero_fusion(tmp_path, mode):
    cfg = config(1)
    baseline = Qwen3DSparkModel(cfg).eval()
    baseline.save_pretrained(tmp_path)
    cfg = side_config(mode, steps=2, block=6)
    model, info = Qwen3DSparkModel.from_pretrained(tmp_path, config=cfg,
        attn_implementation="sdpa", output_loading_info=True)
    assert set(info["missing_keys"]) == {"side_fusion.down.weight", "side_fusion.up.weight"}
    assert not info["unexpected_keys"]
    assert model.side_fusion.up.weight.count_nonzero() == 0
    model.eval()
    with torch.no_grad():
        batch = data(model)
        torch.testing.assert_close(model._forward_backbone(**batch), baseline._forward_backbone(**batch),
                                   atol=3e-6, rtol=3e-5)


def test_group_auxiliary_loss_runs_through_real_trainer(monkeypatch):
    import torch.distributed as dist
    import deepspec.modeling.dspark.qwen3.modeling as modeling
    from deepspec.trainer.dspark_trainer import Qwen3DSparkTrainer
    from deepspec.utils.config import load_config
    monkeypatch.setattr(modeling, "create_dspark_attention_mask", dense_mask)
    monkeypatch.setattr(dist, "get_world_size", lambda: 1)
    model = Qwen3DSparkModel(side_config("grouped", steps=2, block=6)).train()
    model.set_embedding_head_trainable(False)
    args = load_config("config/loop/dflash_flex_qwen3_4b.py")
    trainer = SimpleNamespace(draft_model=model, model=model, args=args,
        global_step=0, next_micro_step=0, global_rank=0, world_size=1,
        gradient_accumulation_steps=1)
    loss = Qwen3DSparkTrainer.run_batch(trainer, dict(input_ids=torch.randint(0, 60, (1, 14)),
        loss_mask=torch.ones(1, 14), target_hidden_states=torch.randn(1, 14, 64)))
    assert torch.isfinite(loss)
    loss.backward()
    assert model.layers[0].self_attn.q_proj.weight.grad.abs().sum() > 0


@pytest.mark.parametrize("mode", ["none", "fusion", "grouped"])
def test_anchor_excluded_labels_mask_and_real_trainer_objective(monkeypatch, mode):
    import torch.distributed as dist
    import deepspec.modeling.dspark.qwen3.modeling as modeling
    from deepspec.trainer.dspark_trainer import Qwen3DSparkTrainer
    from deepspec.utils.config import load_config

    monkeypatch.setattr(modeling, "create_dspark_attention_mask", dense_mask)
    monkeypatch.setattr(dist, "get_world_size", lambda: 1)
    anchors = torch.tensor([[2, 4]])
    monkeypatch.setattr(modeling, "sample_anchor_positions", lambda **kwargs:
                        (anchors, torch.ones_like(anchors, dtype=torch.bool)))
    cfg = side_config(mode, steps=5, block=16)
    cfg.sample_from_anchor = False
    model = Qwen3DSparkModel(cfg).train()
    model.set_embedding_head_trainable(False)
    batch = dict(input_ids=torch.arange(24).unsqueeze(0),
                 loss_mask=torch.ones(1, 24),
                 target_hidden_states=torch.randn(1, 24, 64))
    captured = []
    model.register_forward_hook(lambda module, args, output: captured.append(output))
    args = load_config("config/loop/dflash_flex_qwen3_4b.py")
    trainer = SimpleNamespace(draft_model=model, model=model, args=args,
        global_step=0, next_micro_step=0, global_rank=0, world_size=1,
        gradient_accumulation_steps=1)
    loss = Qwen3DSparkTrainer.run_batch(trainer, batch)
    output = captured[0]
    assert not output.eval_mask[:, :, 0].any()
    assert output.eval_mask[:, :, 1:].all()
    torch.testing.assert_close(output.target_ids,
        anchors.unsqueeze(-1) + torch.arange(16).reshape(1, 1, -1))
    weights = output.eval_mask.float() * torch.exp(-torch.arange(16).float() / 4)
    def weighted_ce(logits):
        ce = F.cross_entropy(logits.flatten(0, 2), output.target_ids.flatten(),
                             reduction="none").reshape_as(weights)
        return (ce * weights).sum() / (weights.sum() + 1e-6)
    expected = weighted_ce(output.draft_logits)
    if mode == "grouped":
        assert output.side_aux_logits.shape == output.draft_logits.shape
        expected = expected + 0.2 * weighted_ce(output.side_aux_logits)
    else:
        assert output.side_aux_logits is None
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert model.layers[0].self_attn.q_proj.weight.grad.abs().sum() > 0
    assert model.lm_head.weight.grad is None


def test_anchor_excluded_group_feedback_is_same_position_across_blocks(monkeypatch):
    cfg = side_config("grouped", steps=2, block=7)
    cfg.sample_from_anchor = False
    model = Qwen3DSparkModel(cfg).eval()
    captured, outputs = [], []
    model.layers[0].register_forward_pre_hook(lambda module, args, kwargs:
        captured.append(kwargs["hidden_states"].detach().clone()), with_kwargs=True)
    model.layers[0].register_forward_hook(lambda module, args, output:
        outputs.append(output.detach().clone()))
    def forced_logits(hidden):
        ids = torch.tensor([[[7, 8, 9], [10, 11, 12]]])
        return hidden.new_zeros(*hidden.shape[:-1], 64).scatter(-1, ids.unsqueeze(-1), 100)
    monkeypatch.setattr(model, "compute_logits", forced_logits)
    with torch.no_grad():
        model._forward_backbone(**data(model, blocks=2))
    second = captured[1].reshape(1, 2, 7, 32)
    torch.testing.assert_close(second[:, :, 1:4],
        model.embed_tokens(torch.tensor([[[7, 8, 9], [10, 11, 12]]])))
    torch.testing.assert_close(second[:, :, 0], model.embed_tokens(torch.tensor([[2, 2]])))
    torch.testing.assert_close(second[:, :, 4:], outputs[0].reshape(1, 2, 7, 32)[:, :, 4:])


@pytest.mark.parametrize("mode", ["none", "fusion", "grouped"])
def test_anchor_excluded_proposal_contains_one_anchor_and_fifteen_predictions(mode):
    from deepspec.eval.dspark.evaluator import Qwen3DSparkEvaluator
    cfg = side_config(mode, block=16)
    cfg.sample_from_anchor = False
    draft = Qwen3DSparkModel(cfg).eval()
    ids = torch.full((1, 16), 63)
    ids[:, 0] = 2
    with torch.no_grad():
        hidden = draft._forward_backbone(**data(draft))
        proposal = build_dspark_proposal(draft, draft_input_ids=ids,
            block_hidden=hidden, block_size=16, temperature=0.0, confidence_threshold=0.0)
    assert proposal.draft_token_count == 15
    assert proposal.verify_input_ids.shape == (1, 16)
    assert proposal.verify_input_ids[0, 0] == 2
    assert proposal.draft_probs.shape == (1, 15, 64)
    evaluator = SimpleNamespace(draft_model=draft)
    assert Qwen3DSparkEvaluator.max_proposal_tokens.fget(evaluator) == 15


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA overlap requires GPU")
@pytest.mark.parametrize("mode", ["fusion", "grouped"])
@pytest.mark.parametrize("dspark", [False, True])
def test_cuda_serial_overlap_hidden_and_cache_equivalence(mode, dspark):
    cfg = side_config(mode, block=16, dspark=dspark)
    cfg.sample_from_anchor = False
    model = Qwen3DSparkModel(cfg).to("cuda", dtype=torch.bfloat16).eval()
    model.profile_side_paths = True
    with torch.no_grad():
        model.side_fusion.up.weight.fill_(0.03)
        cpu = data(model.to("cpu"))
        model.to("cuda")
        batch = {key: value.to("cuda", dtype=torch.bfloat16 if value.is_floating_point() else value.dtype)
                 for key, value in cpu.items()}
        model.side_execution = "serial"
        serial_cache = ContextKVCache()
        serial = model._forward_backbone(**batch, past_key_values=serial_cache)
        model.side_execution = "overlap"
        overlap_cache = ContextKVCache()
        overlap = model._forward_backbone(**batch, past_key_values=overlap_cache)
        torch.cuda.synchronize()
        torch.testing.assert_close(serial, overlap)
        assert model.last_side_timer.resolve()["main_tail_ms"] > 0
        for index in serial_cache.layers:
            for a, b in zip(serial_cache.layers[index], overlap_cache.layers[index]):
                torch.testing.assert_close(a, b)

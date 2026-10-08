import copy
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
from transformers import Qwen3Config, Qwen3ForCausalLM, DynamicCache

from deepspec.modeling.dspark.qwen3.modeling import Qwen3DSparkModel
from deepspec.modeling.dspark.context_cache import ContextKVCache
from deepspec.modeling.dspark.common import extract_context_feature
from deepspec.modeling.dspark.loss import compute_dspark_loss
from deepspec.eval.dspark.draft_ops import forward_dspark_draft_block, build_dspark_proposal
from deepspec.eval.base_evaluator import generate_decoding_sample
from deepspec.utils.metrics import reset


def config(loops=3, dspark=False):
    cfg = Qwen3Config(hidden_size=32, intermediate_size=64, num_hidden_layers=5,
                     num_attention_heads=2, num_key_value_heads=1, head_dim=16,
                     vocab_size=64, max_position_embeddings=128, attention_dropout=0.0)
    cfg.target_layer_ids = [0, 2]
    cfg.mask_token_id = 63
    cfg.block_size = 3
    cfg.num_anchors = 2
    cfg.num_loops = loops
    cfg.loop_loss_weights = [1.0] * loops
    cfg.markov_rank = 8 if dspark else 0
    cfg.markov_head_type = "vanilla"
    cfg.enable_confidence_head = dspark
    cfg.confidence_head_with_markov = dspark
    cfg._attn_implementation = "sdpa"
    return cfg


@pytest.fixture(autouse=True)
def deterministic():
    torch.manual_seed(42)
    torch.set_num_threads(1)
    reset()
    yield
    reset()


def inputs(ctx=6):
    return dict(target_hidden_states=torch.randn(1, ctx, 64),
                noise_embedding=torch.randn(1, 3, 32),
                position_ids=torch.arange(ctx + 3).unsqueeze(0))


def test_one_loop_matches_legacy_and_reuses_only_five_parameter_sets():
    model = Qwen3DSparkModel(config()).eval()
    legacy = Qwen3DSparkModel(config(1)).eval()
    legacy.load_state_dict(model.state_dict())
    batch = inputs()
    expected = legacy._forward_backbone(**batch)
    actual = model._forward_backbone(**batch, num_loops=1)
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
    assert sum(p.numel() for p in model.parameters()) == sum(p.numel() for p in legacy.parameters())
    assert len(model.layers) == 5
    assert len(model._forward_backbone(**batch, return_all_loop_hidden=True)) == 3


@pytest.mark.parametrize("loops", [1, 2, 3])
def test_incremental_context_cache_matches_full_recompute_and_has_no_block_kv(loops):
    model = Qwen3DSparkModel(config()).eval()
    cache = ContextKVCache()
    batch = inputs()
    with torch.no_grad():
        model._forward_backbone(**batch, num_loops=loops, past_key_values=cache)
        assert all(k.shape[-2] == 6 for k, _ in cache.layers.values())
        new_context = torch.randn(1, 2, 64)
        noise = torch.randn(1, 3, 32)
        incremental = model._forward_backbone(
            target_hidden_states=new_context, noise_embedding=noise,
            position_ids=torch.arange(6, 11).unsqueeze(0),
            past_key_values=cache, num_loops=loops)
        expected = model._forward_backbone(
            target_hidden_states=torch.cat((batch["target_hidden_states"], new_context), dim=1),
            noise_embedding=noise, position_ids=torch.arange(11).unsqueeze(0), num_loops=loops)
    torch.testing.assert_close(incremental, expected, atol=3e-6, rtol=3e-5)
    assert len(cache.layers) == 5
    assert all(k.shape[-2] == 8 for k, _ in cache.layers.values())


def test_repeated_loops_reject_legacy_mixed_cache():
    model = Qwen3DSparkModel(config())
    with pytest.raises(ValueError, match="ContextKVCache"):
        model._forward_backbone(**inputs(), past_key_values=DynamicCache())
    for loops in (0, 4):
        with pytest.raises(ValueError, match="num_loops"):
            model._forward_backbone(**inputs(), num_loops=loops)


def dense_mask(*, anchor_positions, block_keep_mask, seq_len, block_size, device):
    # CPU SDPA equivalent of the production flex-attention training mask.
    q = torch.arange(anchor_positions.shape[1] * block_size, device=device)
    k = torch.arange(seq_len + q.numel(), device=device)
    q_blocks = q // block_size
    anchors = anchor_positions[:, q_blocks]
    context = (k < seq_len)[None, None, :] & (k[None, None, :] < anchors[:, :, None])
    block = (k >= seq_len)[None, None, :] & ((k - seq_len)[None, None, :] // block_size == q_blocks[None, :, None])
    return ((context | block) & block_keep_mask[:, q_blocks, None]).unsqueeze(1)


@pytest.mark.parametrize("dspark", [False, True])
def test_multi_exit_training_backward_checkpoint_and_mask(monkeypatch, tmp_path, dspark):
    import deepspec.modeling.dspark.qwen3.modeling as modeling
    monkeypatch.setattr(modeling, "create_dspark_attention_mask", dense_mask)
    model = Qwen3DSparkModel(config(dspark=dspark))
    model.set_embedding_head_trainable(False)
    batch = dict(input_ids=torch.randint(0, 60, (1, 12)), loss_mask=torch.ones(1, 12),
                 target_hidden_states=torch.randn(1, 12, 64),
                 target_last_hidden_states=torch.randn(1, 12, 32))
    exits = model(**batch)
    assert len(exits) == 3
    for output in exits:
        assert output.draft_logits.shape == (1, 2, 3, 64)
        assert torch.equal(output.target_ids, exits[0].target_ids)
    loss = sum(torch.nn.functional.cross_entropy(o.draft_logits.flatten(0, 2), o.target_ids.flatten()) for o in exits)
    loss.backward()
    assert all(layer.self_attn.q_proj.weight.grad is not None for layer in model.layers)
    assert model.fc.weight.grad is not None and model.fc.weight.grad.abs().sum() > 0
    assert model.embed_tokens.weight.grad is None
    assert model.lm_head.weight.grad is None
    if dspark:
        assert model.markov_head.markov_w2.weight.grad is not None
    model.save_pretrained(tmp_path)
    restored = Qwen3DSparkModel.from_pretrained(tmp_path, attn_implementation="sdpa")
    assert restored.num_loops == 3 and len(restored.layers) == 5
    # Future target features must not leak into an anchor block, at any loop.
    anchors = torch.tensor([[4]])
    keep = torch.tensor([[True]])
    hidden = torch.randn(1, 10, 64)
    noise = torch.randn(1, 3, 32)
    kwargs = dict(noise_embedding=noise, position_ids=torch.cat((torch.arange(10), torch.arange(4, 7))).unsqueeze(0),
                  attention_mask=dense_mask(anchor_positions=anchors, block_keep_mask=keep, seq_len=10, block_size=3, device=noise.device),
                  return_all_loop_hidden=True)
    before = model._forward_backbone(target_hidden_states=hidden, **kwargs)
    changed = hidden.clone(); changed[:, 4:] = torch.randn_like(changed[:, 4:]) * 100
    after = model._forward_backbone(target_hidden_states=changed, **kwargs)
    for a, b in zip(before, after):
        torch.testing.assert_close(a, b)


@pytest.mark.parametrize("loops", [1, 2, 3])
def test_speculative_greedy_generation_matches_target_ar(loops):
    target_cfg = copy.deepcopy(config(1))
    target_cfg.num_hidden_layers = 4
    target_cfg.layer_types = ["full_attention"] * 4
    target = Qwen3ForCausalLM(target_cfg).eval()
    draft = Qwen3DSparkModel(config()).eval()
    draft.eval_num_loops = loops
    prompt = torch.tensor([[2, 7, 9, 3]])
    def init_context(initial_output, **kwargs):
        return SimpleNamespace(past_key_values_draft=ContextKVCache(),
            target_hidden_states=extract_context_feature(initial_output.hidden_states, draft.target_layer_ids))
    def propose(context, output_ids, position_ids, start, stop_token_ids):
        ids = torch.full((1, 3), 63); ids[:, 0] = output_ids[:, start]
        hidden = forward_dspark_draft_block(draft, draft_input_ids=ids, position_ids=position_ids,
            past_key_values_draft=context.past_key_values_draft, target_hidden_states=context.target_hidden_states,
            start=start, block_size=3)
        return build_dspark_proposal(draft, draft_input_ids=ids, block_hidden=hidden, block_size=3,
            temperature=0.0, confidence_threshold=0.0)
    def update(context, verification):
        context.target_hidden_states = extract_context_feature(verification.target_output.hidden_states,
            draft.target_layer_ids)[:, :verification.accepted_draft_tokens + 1]
    result = generate_decoding_sample(target_model=target, input_ids=prompt, max_new_tokens=12,
        max_proposal_tokens=3, temperature=0.0, stop_token_ids=None,
        init_context=init_context, propose=propose, update=update, profile=True)
    expected = prompt.clone()
    with torch.no_grad():
        for _ in range(12):
            token = target(expected).logits[:, -1].argmax(-1, keepdim=True)
            expected = torch.cat((expected, token), dim=1)
    assert torch.equal(result.output_ids, expected)
    assert result.decode_wall_ms > 0 and result.draft_ms > 0


def test_real_multi_exit_distillation_and_confidence_loss_backward(monkeypatch):
    import deepspec.modeling.dspark.qwen3.modeling as modeling
    from deepspec.utils import metrics
    monkeypatch.setattr(modeling, "create_dspark_attention_mask", dense_mask)
    # Exercise the real single-rank loss without opening distributed sockets.
    monkeypatch.setattr(dist, "get_world_size", lambda: 1)
    model = Qwen3DSparkModel(config(dspark=True))
    model.set_embedding_head_trainable(False)
    outputs = model(input_ids=torch.randint(0, 60, (1, 12)), loss_mask=torch.ones(1, 12),
        target_hidden_states=torch.randn(1, 12, 64), target_last_hidden_states=torch.randn(1, 12, 32))
    loss = sum(weight * compute_dspark_loss(outputs=output, loss_decay_gamma=4.0,
        ce_loss_alpha=0.1, l1_loss_alpha=0.9, confidence_head_alpha=1.0,
        metric_tag=f"train/loop{index + 1}")
        for index, (weight, output) in enumerate(zip((0.2, 0.3, 0.5), outputs)))
    assert torch.isfinite(loss)
    loss.backward()
    assert model.confidence_head.proj.weight.grad.abs().sum() > 0
    assert model.fc.weight.grad.abs().sum() > 0
    assert all(f"train/loop{index}/ce_loss" in metrics._metrics for index in (1, 2, 3))
    assert all(f"train/loop{index}/confidence_loss" in metrics._metrics for index in (1, 2, 3))


def test_eval_task_selection_and_ar_flags(monkeypatch):
    import eval as entrypoint
    monkeypatch.setattr("sys.argv", ["eval.py", "--target_name_or_path", "target",
        "--draft_name_or_path", "draft", "--tasks", "gsm8k,mbpp", "--max-samples", "2", "--num-loops", "3"])
    args = entrypoint.parse_args()
    assert args.tasks == [("gsm8k", 2), ("mbpp", 2)] and args.num_loops == 3
    monkeypatch.setattr("sys.argv", ["eval.py", "--target_name_or_path", "target", "--autoregressive"])
    args = entrypoint.parse_args()
    assert args.autoregressive and args.draft_name_or_path == "AR"


def test_loop_config_overrides_and_cache_feature_count_independent_of_depth():
    from deepspec.utils.config import load_config, parse_opts_to_config
    from deepspec.modeling.dspark.qwen3.config import build_draft_config
    for variant, depth, loops in (("5l", 5, 1), ("15l", 15, 1), ("loop5x3", 5, 3)):
        args = parse_opts_to_config(["run_root=/tmp/loop-runs", "exp_name=example"],
            load_config(f"config/loop/dflash_{variant}_qwen3_4b.py"))
        assert args.logging.checkpoint_dir == "/tmp/loop-runs/deepspec/example"
        target = Qwen3Config(num_hidden_layers=36)
        draft_cfg = build_draft_config(target, args.model)
        assert draft_cfg.num_hidden_layers == depth and draft_cfg.num_loops == loops
        assert draft_cfg.target_layer_ids == [1, 9, 17, 25, 33]

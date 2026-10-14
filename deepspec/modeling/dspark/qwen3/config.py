import copy
import math

from deepspec.modeling.dspark.common import validate_target_layer_ids


TRAIN_ATTN_IMPLEMENTATION = "flex_attention"


def build_draft_config(
    target_config,
    model_args,
):
    num_target_layers = int(target_config.num_hidden_layers)
    num_draft_layers = int(model_args.num_draft_layers)
    if num_draft_layers < 1:
        raise ValueError("num_draft_layers must be >= 1")
    layer_types = ["full_attention"] * num_draft_layers
    assert "target_layer_ids" in model_args, "target_layer_ids must be provided."
    target_layer_ids = validate_target_layer_ids(
        model_args.target_layer_ids,
        num_target_layers,
    )

    confidence_head_alpha = float(model_args.confidence_head_alpha)
    assert confidence_head_alpha >= 0.0
    enable_confidence_head = confidence_head_alpha > 0.0
    if enable_confidence_head:
        assert "confidence_head_with_markov" in model_args, (
            "confidence_head_with_markov must be provided when "
            "confidence_head_alpha > 0."
        )
    markov_rank = int(model_args.markov_rank)
    assert markov_rank >= 0, f"markov_rank must be >= 0, got {markov_rank}"
    if markov_rank > 0:
        assert "markov_head_type" in model_args, (
            "markov_head_type must be provided when markov_rank > 0."
        )

    draft_config = copy.deepcopy(target_config)
    draft_config.architectures = ["Qwen3DSparkModel"]
    draft_config.num_target_layers = num_target_layers
    draft_config.num_hidden_layers = num_draft_layers
    draft_config.num_loops = int(model_args.get("num_loops", 1))
    if draft_config.num_loops < 1:
        raise ValueError("num_loops must be >= 1")
    draft_config.loop_loss_weights = list(model_args.get(
        "loop_loss_weights", [1.0] * draft_config.num_loops
    ))
    if len(draft_config.loop_loss_weights) != draft_config.num_loops:
        raise ValueError("loop_loss_weights must have one entry per loop")
    if any(not math.isfinite(float(w)) or float(w) < 0 for w in draft_config.loop_loss_weights) or sum(draft_config.loop_loss_weights) <= 0:
        raise ValueError("loop_loss_weights must be finite, nonnegative, with positive sum")
    if model_args.get("sample_loop_count", False) and draft_config.loop_loss_weights[0] <= 0:
        raise ValueError("sample_loop_count requires a positive first loop loss weight")
    draft_config.loop_boundary_norm = bool(model_args.get("loop_boundary_norm", False))
    draft_config.enable_exit_gate = bool(model_args.get("enable_exit_gate", False))
    draft_config.ouro_training_mode = str(model_args.get("ouro_training_mode", "baseline"))
    draft_config.min_exit_loop = int(model_args.get("min_exit_loop", 3))
    draft_config.ouro_beta = float(model_args.get("ouro_beta", 0.1))
    draft_config.gate_gain_slope = float(model_args.get("gate_gain_slope", 50.0))
    draft_config.gate_gain_threshold = float(model_args.get("gate_gain_threshold", 0.005))
    draft_config.exit_gate_trained = False
    from deepspec.modeling.dspark.ouro import validate_exit_config
    validate_exit_config(draft_config)
    draft_config.block_size = int(model_args.block_size)
    draft_config.tie_word_embeddings = False
    draft_config.layer_types = layer_types
    draft_config._attn_implementation = TRAIN_ATTN_IMPLEMENTATION
    draft_config.mask_token_id = int(model_args.mask_token_id)
    draft_config.target_layer_ids = target_layer_ids
    draft_config.num_anchors = int(model_args.num_anchors)
    draft_config.enable_confidence_head = enable_confidence_head
    if enable_confidence_head:
        draft_config.confidence_head_with_markov = bool(
            model_args.confidence_head_with_markov
        )
    draft_config.markov_rank = markov_rank
    if markov_rank > 0:
        draft_config.markov_head_type = str(model_args.markov_head_type)
    from deepspec.modeling.dspark.side_loop import SIDE_CONFIG_DEFAULTS, validate_side_config
    for key, default in SIDE_CONFIG_DEFAULTS.items():
        setattr(draft_config, key, model_args.get(key, default))
    validate_side_config(draft_config)
    return draft_config


__all__ = [
    "build_draft_config",
]


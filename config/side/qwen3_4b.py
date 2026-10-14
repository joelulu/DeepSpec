"""Common five-layer, anchor-excluded side experiments; launcher selects heads."""
import os

from deepspec.trainer import Qwen3DSparkTrainer
from deepspec.utils.constant import BASE_CKPT_DIR, BASE_TB_DIR, QWEN_3_4B

project_name = "deepspec"
exp_name = "dflash_side_qwen3_4b"
seed = 42
run_root = None

model = dict(
    algorithm="dflash", target_model_name_or_path=QWEN_3_4B,
    block_size=16, sample_from_anchor=False, num_draft_layers=5,
    num_loops=1, loop_loss_weights=[1.0], loop_loss_mode="final",
    loop_boundary_norm=False, sample_loop_count=False,
    side_loop_mode="none", side_loop_steps=5, side_group_size=3,
    side_fusion_rank=128, side_aux_loss_alpha=0.2,
    allow_block_size_transfer=False, allow_prediction_convention_transfer=False,
    init_checkpoint=None,
    target_layer_ids=[1, 9, 17, 25, 33], mask_token_id=151669, num_anchors=32,
    markov_rank=0, markov_head_type="vanilla", confidence_head_with_markov=False,
    confidence_head_alpha=0.0,
    loss_decay_gamma=4.0, ce_loss_alpha=1.0, l1_loss_alpha=0.0,
)
train = dict(
    trainer_cls=Qwen3DSparkTrainer, lr=6e-4, warmup_ratio=0.04,
    weight_decay=0.0, precision="bf16", local_batch_size=2, global_batch_size=64,
    num_train_epochs=3, max_train_steps=None, max_grad_norm=1.0,
    sharding_strategy="no_shard", torch_compile=False,
)
logging = dict(logging_steps=5, diagnostic_steps=0, checkpointing_steps=100,
               save_only_final=True, save_training_state=False)
data = dict(target_cache_path=None, data_percent=3, holdout_samples=32,
            chat_template="qwen", max_length=4096, num_workers=4)


def finalize_cfg(cfg):
    logging_cfg = dict(cfg["logging"])
    logging_cfg["checkpoint_dir"] = os.path.join(
        cfg.get("run_root") or BASE_CKPT_DIR, cfg["project_name"], cfg["exp_name"])
    tb_root = os.path.join(cfg["run_root"], "tensorboard") if cfg.get("run_root") else BASE_TB_DIR
    logging_cfg["tensorboard_dir"] = os.path.join(tb_root, cfg["project_name"], cfg["exp_name"])
    cfg["logging"] = logging_cfg
    return cfg

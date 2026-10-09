import os

from deepspec.trainer import Qwen3DSparkTrainer
from deepspec.utils.constant import BASE_CKPT_DIR, BASE_TB_DIR, QWEN_3_4B

project_name = "deepspec"
exp_name = "dflash_flex_qwen3_4b"
seed = 42
run_root = None

model = dict(
    target_model_name_or_path=QWEN_3_4B,
    block_size=7,
    num_draft_layers=5,
    num_loops=1,
    loop_loss_weights=[1.0],
    loop_loss_mode="final",
    loop_boundary_norm=False,
    sample_loop_count=False,
    target_layer_ids=[1, 9, 17, 25, 33],
    mask_token_id=151669,
    num_anchors=32,

    # Disable markov head.
    markov_rank=0,

    # Disable confidence head.
    confidence_head_alpha=0.0,

    # CE-only loss.
    loss_decay_gamma=4.0,
    ce_loss_alpha=1.0,
    l1_loss_alpha=0.0,
)

train = dict(
    trainer_cls=Qwen3DSparkTrainer,
    lr=6.0e-4,
    warmup_ratio=0.04,
    weight_decay=0.0,
    precision="bf16",
    local_batch_size=4,
    global_batch_size=256,
    num_train_epochs=3,
    max_train_steps=None,
    max_grad_norm=1.0,
    sharding_strategy="no_shard",
    torch_compile=False,
)

logging = dict(
    logging_steps=5,
    diagnostic_steps=20,
    checkpointing_steps=100,
    save_only_final=True,
    save_training_state=False,
)

data = dict(
    target_cache_path=None,
    data_percent=3,
    holdout_samples=32,
    chat_template="qwen",
    max_length=4096,
    num_workers=4,
)


def finalize_cfg(cfg):
    logging_cfg = dict(cfg["logging"])
    project_name=str(cfg['project_name'])
    exp_name = str(cfg["exp_name"])
    logging_cfg["checkpoint_dir"] = os.path.join(cfg.get("run_root") or BASE_CKPT_DIR, project_name, exp_name)
    logging_cfg["tensorboard_dir"] = os.path.join(os.path.join(cfg["run_root"], "tensorboard") if cfg.get("run_root") else BASE_TB_DIR, project_name, exp_name)
    cfg["logging"] = logging_cfg

    return cfg


# Shared, fixed settings for the 3% pilot. Edit this file to change paths/budget.
# Source only after changing to the repository root.
export CUDA_VISIBLE_DEVICES=0,1,2,3
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29510
export RANK=0
export WORLD_SIZE=1
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=offline
export SOURCE_CACHE=/home/jovyan/LMM/gaohl/project/from_aliyun/LDM/dllm_dflash/DeepSpec-main/cache/qwen3_4b_target_cache_130w_20260713_041400
export TARGET_CACHE="$PWD/cache/qwen3_4b_130w_3pct_seed42"
export GLOBAL_BATCH=256
export LOCAL_BATCH=1
export STEPS=1000
export SAVE_EVERY=500
export SEED=42
export ANCHORS=32
export LR=6e-4
export SHARDING=no_shard
export RUN_ROOT="$PWD/runs/dflash_loop_3pct"
export TASKS=gsm8k,mbpp,alpaca
export MAX_SAMPLES=64
export MAX_NEW_TOKENS=512
export TEMPERATURE=0
export WARMUP_SAMPLES=1
# Offline diagnostic: up to 40 rounds per sample; median of 3 probes.
export PROBE_SAMPLES=16
export PROBE_ROUNDS=40
export PROBE_REPEATS=3

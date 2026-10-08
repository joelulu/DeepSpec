#!/usr/bin/env bash
set -euo pipefail

# Run from this repository, regardless of the caller's working directory.
repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$repo_root"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

variant=${1:?Usage: bash train_dflash_loop.sh 5l|15l|loop5x3 [--opts key=value ...]}
shift
case "$variant" in
  5l|15l|loop5x3) ;;
  *) echo "Unknown DFlash variant: $variant" >&2; exit 2 ;;
esac

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
export MASTER_PORT=${MASTER_PORT:-29510}
export RANK=${RANK:-0}
export WORLD_SIZE=${WORLD_SIZE:-1}
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=offline

export TARGET_CACHE=${TARGET_CACHE:-${target_cache_dir:-/home/jovyan/LMM/gaohl/project/from_aliyun/LDM/dllm_dflash/DeepSpec-main/cache/qwen3_4b_target_cache_50k_20260707_145641}}
export GLOBAL_BATCH=${GLOBAL_BATCH:-256}
export LOCAL_BATCH=${LOCAL_BATCH:-1}
export STEPS=${STEPS:-1000}
export SAVE_EVERY=${SAVE_EVERY:-500}
export SEED=${SEED:-42}
export ANCHORS=${ANCHORS:-32}
export RUN_ROOT=${RUN_ROOT:-$PWD/runs/dflash_loop_1008}
export EXP_NAME=${EXP_NAME:-dflash_4b_${variant}_loop_pilot_seed${SEED}}

echo "Using target cache: $TARGET_CACHE"
echo "Visible GPUs: $CUDA_VISIBLE_DEVICES"
echo "Variant: $variant; batch: $GLOBAL_BATCH; steps: $STEPS; anchors: $ANCHORS"
echo "Experiment: $EXP_NAME"
echo "Checkpoint root: $RUN_ROOT"
echo "[START] $(date '+%Y-%m-%d %H:%M:%S')"

# The shared launcher reads target model/layer IDs from the cache manifest,
# runs preflight, and then starts train.py (which spawns the GPU workers).
bash scripts/loop/train.sh dflash "$variant" \
  --opts "train.torch_compile=False" \
  --opts "logging.logging_steps=5" \
  "$@"

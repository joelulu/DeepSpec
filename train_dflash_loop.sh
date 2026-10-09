#!/usr/bin/env bash
set -euo pipefail

# Run from this repository, regardless of the caller's working directory.
repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$repo_root"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

if [[ $# -ne 1 ]]; then
  echo "Usage: bash train_dflash_loop.sh 5l|15l|loop5x3" >&2; exit 2
fi
variant="$1"
case "$variant" in
  5l|15l|loop5x3) ;;
  *) echo "Unknown DFlash variant: $variant" >&2; exit 2 ;;
esac

source scripts/loop/settings_3pct.sh
unset TARGET_MODEL
export EXP_NAME="dflash_4b_${variant}_3pct_seed${SEED}"
python scripts/loop/cache_subset.py --source "$SOURCE_CACHE" --output "$TARGET_CACHE" \
  --fraction 0.03 --seed "$SEED"

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
  --opts "logging.logging_steps=5"

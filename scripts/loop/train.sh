#!/usr/bin/env bash
set -euo pipefail
repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$repo_root"
export PYTHONPATH="$repo_root${PYTHONPATH:+:$PYTHONPATH}"
python_bin=${PYTHON:-python}
algorithm=${1:?Usage: bash scripts/loop/train.sh dflash|dspark 5l|15l|loop5x3 [--opts key=value ...]}
variant=${2:?Missing variant: 5l|15l|loop5x3}
shift 2
case "$algorithm:$variant" in
  dflash:5l|dflash:15l|dflash:loop5x3|dspark:5l|dspark:15l|dspark:loop5x3) ;;
  *) echo "Unknown algorithm/variant: $algorithm/$variant" >&2; exit 2 ;;
esac
: "${TARGET_CACHE:?Set TARGET_CACHE to your existing DeepSpec cache directory}"
if [[ -z ${TARGET_MODEL:-} ]]; then
  TARGET_MODEL=$("$python_bin" -c 'import json,sys; print(json.load(open(sys.argv[1]))["target_model_name_or_path"])' "$TARGET_CACHE/manifest.json")
fi
target_layers=$("$python_bin" -c 'import json,sys; print(json.dumps(json.load(open(sys.argv[1]))["target_layer_ids"]))' "$TARGET_CACHE/manifest.json")
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
steps=${STEPS:-500}
global_batch=${GLOBAL_BATCH:-16}
local_batch=${LOCAL_BATCH:-1}
run_root=${RUN_ROOT:-$repo_root/runs/loop}
exp_name=${EXP_NAME:-${algorithm}_${variant}_seed${SEED:-42}}
"$python_bin" scripts/loop/preflight.py --cache "$TARGET_CACHE" --target "$TARGET_MODEL" \
  --global-batch "$global_batch" --local-batch "$local_batch" --check-cuda
"$python_bin" train.py --config "config/loop/${algorithm}_${variant}_qwen3_4b.py" \
  --opts "model.target_model_name_or_path='$TARGET_MODEL'" \
  --opts "model.target_layer_ids=$target_layers" \
  --opts "data.target_cache_path='$TARGET_CACHE'" \
  --opts "run_root='$run_root'" \
  --opts "exp_name='$exp_name'" \
  --opts "seed=${SEED:-42}" \
  --opts "model.num_anchors=${ANCHORS:-32}" \
  --opts "train.global_batch_size=$global_batch" \
  --opts "train.local_batch_size=$local_batch" \
  --opts "train.max_train_steps=$steps" \
  --opts "train.lr=${LR:-6e-4}" \
  --opts "train.sharding_strategy=${SHARDING:-no_shard}" \
  --opts "logging.checkpointing_steps=${SAVE_EVERY:-100}" \
  "$@"

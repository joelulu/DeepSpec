#!/usr/bin/env bash
set -euo pipefail
repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$repo_root"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
: "${TARGET_MODEL:?Set TARGET_MODEL to the Qwen3 target used by this checkpoint}"
checkpoint=${1:?Usage: bash eval_dflash_loopcd.sh CHECKPOINT [LOOPS]}
loops=${2:-5}
args=(--target "$TARGET_MODEL" --draft "$checkpoint" --num-loops "$loops"
  --tasks "${TASKS:-gsm8k}" --max-samples "${MAX_SAMPLES:-16}"
  --max-new-tokens "${MAX_NEW_TOKENS:-512}" --temperature "${TEMPERATURE:-0}"
  --seed "${SEED:-42}" --warmup-samples "${WARMUP_SAMPLES:-1}"
  --repeats "${REPEATS:-3}"
  --output-dir "${RESULT_DIR:-$PWD/runs/loopcd/$(date +%Y%m%d_%H%M%S)}")
if [[ ${DRY_RUN:-0} == 1 ]]; then args+=(--dry-run); fi
"${PYTHON:-python}" scripts/loop/loopcd_sweep.py "${args[@]}"

#!/usr/bin/env bash
set -euo pipefail
repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$repo_root"
export PYTHONPATH="$repo_root${PYTHONPATH:+:$PYTHONPATH}"
python_bin=${PYTHON:-python}
checkpoint=${1:?Usage: bash scripts/loop/eval.sh CHECKPOINT [LOOPS]}
loops=${2:-1}
: "${TARGET_MODEL:?Set TARGET_MODEL to the target used by this checkpoint}"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
output=${RESULT_JSON:-$checkpoint/eval_loop${loops}.json}
"$python_bin" eval.py --target_name_or_path "$TARGET_MODEL" --draft_name_or_path "$checkpoint" \
  --tasks "${TASKS:-gsm8k,mbpp,alpaca}" --max-samples "${MAX_SAMPLES:-64}" \
  --max-new-tokens "${MAX_NEW_TOKENS:-512}" --temperature "${TEMPERATURE:-0}" \
  --confidence-threshold 0 --num-loops "$loops" --profile \
  --warmup-samples "${WARMUP_SAMPLES:-1}" --output-json "$output"

#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
source scripts/loop/settings.sh
export CUDA_VISIBLE_DEVICES=${EVAL_GPU:-0}
if [[ $CUDA_VISIBLE_DEVICES == *,* ]]; then
  echo "EVAL_GPU must select a single GPU for serial timing" >&2; exit 2
fi
python_bin=${PYTHON:-python}
if [[ ! $EVAL_REPEATS =~ ^[1-9][0-9]*$ ]]; then
  echo "EVAL_REPEATS must be a positive integer" >&2; exit 2
fi
TARGET_MODEL=$("$python_bin" -c 'import json,sys; print(json.load(open(sys.argv[1]))["target_model_name_or_path"])' "$SOURCE_CACHE/manifest.json")
percent=$("$python_bin" -c 'from decimal import Decimal; import sys; print(format(Decimal(sys.argv[1]).normalize(), "f"))' "$DATA_PERCENT")
# Pass explicit checkpoint paths to test old runs or only one newly trained model.
checkpoints=("$@")
if [[ ${#checkpoints[@]} == 0 ]]; then
  for group in '5 1 raw' '15 1 raw' '1 5 raw' '5 3 raw' '5 3 rmsnorm'; do
    read -r layers loops transfer <<< "$group"
    checkpoints+=("$RUN_ROOT/deepspec/dflash_L${layers}_R${loops}_p${percent}_e${EPOCHS}_${transfer}_${LOOP_LOSS}_seed${SEED}/step_latest")
  done
fi
for checkpoint in "${checkpoints[@]}"; do
  test -f "$checkpoint/config.json" || { echo "Missing checkpoint: $checkpoint" >&2; exit 1; }
done
# A fresh directory prevents old JSON from contaminating this comparison.
mkdir -p "$RUN_ROOT/results"
OUT=$(mktemp -d "$RUN_ROOT/results/temp_$(date +%Y%m%d_%H%M%S)_XXXXXX")
COMMON_ARGS=(--target_name_or_path "$TARGET_MODEL" --tasks "$TASKS"
  --max-samples "$MAX_SAMPLES" --max-new-tokens "$MAX_NEW_TOKENS"
  --temperature 0 --seed "$SEED" --profile --warmup-samples "$WARMUP_SAMPLES")
for ((repeat=1; repeat<=EVAL_REPEATS; repeat++)); do
  result="$OUT/repeat${repeat}"
  "$python_bin" eval.py --autoregressive "${COMMON_ARGS[@]}" --output-json "$result/ar/eval_loop0.json"
  for index in "${!checkpoints[@]}"; do
    checkpoint=${checkpoints[$index]}
    max_loops=$("$python_bin" -c 'import json,sys; print(json.load(open(sys.argv[1])).get("num_loops",1))' "$checkpoint/config.json")
    for ((loops=1; loops<=max_loops; loops++)); do
      "$python_bin" eval.py "${COMMON_ARGS[@]}" --draft_name_or_path "$checkpoint" \
        --num-loops "$loops" --confidence-threshold 0 \
        --output-json "$result/model${index}/eval_loop${loops}.json"
    done
  done
  "$python_bin" scripts/loop/summarize.py "$result" --output "$result/comparison.csv"
  cat "$result/comparison.csv"
done
echo "Results: $OUT"

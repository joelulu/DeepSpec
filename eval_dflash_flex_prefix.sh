#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
checkpoint=${1:?Usage: bash eval_dflash_flex_prefix.sh CHECKPOINT}
source scripts/loop/settings.sh
export CUDA_VISIBLE_DEVICES=${EVAL_GPU:-0}
if [[ $CUDA_VISIBLE_DEVICES == *,* ]]; then
  echo "EVAL_GPU must select a single GPU for serial timing" >&2; exit 2
fi
python_bin=${PYTHON:-python}
TARGET_MODEL=$("$python_bin" -c 'import json,sys; print(json.load(open(sys.argv[1]))["target_model_name_or_path"])' "$SOURCE_CACHE/manifest.json")
mkdir -p "$RUN_ROOT/results"
OUT=$(mktemp -d "$RUN_ROOT/results/prefix_XXXXXX")
"$python_bin" scripts/loop/same_prefix.py --target "$TARGET_MODEL" --draft "$checkpoint" \
  --tasks "$TASKS" --max-samples "$PROBE_SAMPLES" --max-rounds "$PROBE_ROUNDS" \
  --max-new-tokens "$MAX_NEW_TOKENS" --repeats "$PROBE_REPEATS" --seed "$SEED" \
  --warmup-samples "$WARMUP_SAMPLES" --output "$OUT/same_prefix.jsonl"
"$python_bin" scripts/loop/summarize_prefix.py "$OUT/same_prefix.jsonl" --output "$OUT/summary.json"

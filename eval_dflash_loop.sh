#!/usr/bin/env bash
set -euo pipefail
repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$repo_root"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
if [[ $# -ne 1 ]]; then
  echo "Usage: bash eval_dflash_loop.sh fixed|prefix" >&2; exit 2
fi
case "$1" in fixed|prefix) ;; *) echo "Unknown evaluation: $1" >&2; exit 2 ;; esac
source scripts/loop/settings_3pct.sh
# Use one GPU for serial timing in every comparison group.
export CUDA_VISIBLE_DEVICES=0
TARGET_MODEL=$(python -c 'import json,os; print(json.load(open(os.path.join(os.environ["SOURCE_CACHE"],"manifest.json")))["target_model_name_or_path"])')
export TARGET_MODEL
checkpoint_root="$RUN_ROOT/deepspec"
mkdir -p "$RUN_ROOT/results"
if [[ "$1" == fixed ]]; then
  for variant in 5l 15l loop5x3; do
    test -f "$checkpoint_root/dflash_4b_${variant}_3pct_seed42/step_latest/config.json" || {
      echo "Missing checkpoint for $variant; train this group first" >&2; exit 1;
    }
  done
  python eval.py --autoregressive --target_name_or_path "$TARGET_MODEL" \
    --tasks "$TASKS" --max-samples "$MAX_SAMPLES" --max-new-tokens "$MAX_NEW_TOKENS" \
    --temperature 0 --seed "$SEED" --profile --warmup-samples "$WARMUP_SAMPLES" \
    --output-json "$RUN_ROOT/results/ar/eval_loop0.json"
  for variant in 5l 15l loop5x3; do
    checkpoint="$checkpoint_root/dflash_4b_${variant}_3pct_seed42/step_latest"
    loops_list=1
    if [[ "$variant" == loop5x3 ]]; then loops_list="1 2 3"; fi
    for loops in $loops_list; do
      python eval.py --target_name_or_path "$TARGET_MODEL" --draft_name_or_path "$checkpoint" \
        --tasks "$TASKS" --max-samples "$MAX_SAMPLES" --max-new-tokens "$MAX_NEW_TOKENS" \
        --temperature 0 --seed "$SEED" --confidence-threshold 0 --num-loops "$loops" \
        --profile --warmup-samples "$WARMUP_SAMPLES" \
        --output-json "$RUN_ROOT/results/$variant/eval_loop${loops}.json"
    done
  done
  python scripts/loop/summarize.py "$RUN_ROOT/results" --output "$RUN_ROOT/comparison.csv"
else
  python scripts/loop/same_prefix.py --target "$TARGET_MODEL" \
    --draft "$checkpoint_root/dflash_4b_loop5x3_3pct_seed42/step_latest" \
    --tasks "$TASKS" --max-samples "$PROBE_SAMPLES" --max-rounds "$PROBE_ROUNDS" \
    --max-new-tokens "$MAX_NEW_TOKENS" --repeats "$PROBE_REPEATS" --seed "$SEED" \
    --warmup-samples "$WARMUP_SAMPLES" --output "$RUN_ROOT/same_prefix.jsonl"
  python scripts/loop/summarize_prefix.py "$RUN_ROOT/same_prefix.jsonl" \
    --output "$RUN_ROOT/same_prefix_summary.json"
fi

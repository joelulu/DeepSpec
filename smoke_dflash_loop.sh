#!/usr/bin/env bash
set -euo pipefail
repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$repo_root"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
if [[ $# -ne 0 ]]; then echo "Usage: bash smoke_dflash_loop.sh" >&2; exit 2; fi
source scripts/loop/settings_3pct.sh
export RUN_ROOT="$PWD/runs/dflash_loop_3pct_smoke"
export STEPS=2 SAVE_EVERY=2 GLOBAL_BATCH=4 ANCHORS=2
unset TARGET_MODEL
python scripts/loop/cache_subset.py --source "$SOURCE_CACHE" --output "$TARGET_CACHE" \
  --fraction 0.03 --seed "$SEED"
for variant in 5l 15l loop5x3; do
  export EXP_NAME="dflash_4b_${variant}_3pct_smoke_seed42"
  bash scripts/loop/train.sh dflash "$variant" \
    --opts "train.torch_compile=False" --opts "logging.logging_steps=1"
done
export CUDA_VISIBLE_DEVICES=0
TARGET_MODEL=$(python -c 'import json,os; print(json.load(open(os.path.join(os.environ["SOURCE_CACHE"],"manifest.json")))["target_model_name_or_path"])')
export TARGET_MODEL
checkpoint_root="$RUN_ROOT/deepspec"
python eval.py --autoregressive --target_name_or_path "$TARGET_MODEL" --tasks gsm8k \
  --max-samples 2 --max-new-tokens 32 --temperature 0 --seed "$SEED" --profile \
  --warmup-samples 1 --output-json "$RUN_ROOT/results/ar/eval_loop0.json"
for variant in 5l 15l loop5x3; do
  checkpoint="$checkpoint_root/dflash_4b_${variant}_3pct_smoke_seed42/step_latest"
  loops_list=1
  if [[ "$variant" == loop5x3 ]]; then loops_list="1 2 3"; fi
  for loops in $loops_list; do
    python eval.py --target_name_or_path "$TARGET_MODEL" --draft_name_or_path "$checkpoint" \
      --tasks gsm8k --max-samples 2 --max-new-tokens 32 --temperature 0 --seed "$SEED" \
      --num-loops "$loops" --confidence-threshold 0 --profile --warmup-samples 1 \
      --output-json "$RUN_ROOT/results/$variant/eval_loop${loops}.json"
  done
done
python scripts/loop/same_prefix.py --target "$TARGET_MODEL" \
  --draft "$checkpoint_root/dflash_4b_loop5x3_3pct_smoke_seed42/step_latest" \
  --tasks gsm8k --max-samples 2 --max-rounds 2 --max-new-tokens 32 --repeats 1 \
  --seed "$SEED" --warmup-samples 1 --output "$RUN_ROOT/same_prefix.jsonl"
python scripts/loop/summarize.py "$RUN_ROOT/results" --output "$RUN_ROOT/comparison.csv"
python scripts/loop/summarize_prefix.py "$RUN_ROOT/same_prefix.jsonl" --output "$RUN_ROOT/same_prefix_summary.json"
echo "GPU smoke complete. Checkpoints are isolated from the 1000-step pilot."

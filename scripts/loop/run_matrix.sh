#!/usr/bin/env bash
set -euo pipefail
repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$repo_root"
python_bin=${PYTHON:-python}
: "${TARGET_CACHE:?Set TARGET_CACHE to your existing DeepSpec cache directory}"
if [[ -z ${TARGET_MODEL:-} ]]; then
  TARGET_MODEL=$("$python_bin" -c 'import json,sys; print(json.load(open(sys.argv[1]))["target_model_name_or_path"])' "$TARGET_CACHE/manifest.json")
fi
export TARGET_MODEL
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
run_root=${RUN_ROOT:-$repo_root/runs/loop}
if [[ ${RUN_AR:-1} == 1 ]]; then
  "$python_bin" eval.py --autoregressive --target_name_or_path "$TARGET_MODEL" \
    --tasks "${TASKS:-gsm8k,mbpp,alpaca}" --max-samples "${MAX_SAMPLES:-64}" \
    --max-new-tokens "${MAX_NEW_TOKENS:-512}" --temperature "${TEMPERATURE:-0}" --profile \
    --warmup-samples "${WARMUP_SAMPLES:-1}" --output-json "$run_root/ar/eval_loop0.json"
fi
algorithms=${ALGORITHMS:-dflash}
variants=${VARIANTS:-5l 15l loop5x3}
for algorithm in $algorithms; do
  for variant in $variants; do
    EXP_NAME="${algorithm}_${variant}_seed${SEED:-42}" bash scripts/loop/train.sh "$algorithm" "$variant"
    checkpoint="$run_root/deepspec/${algorithm}_${variant}_seed${SEED:-42}/step_latest"
    if [[ "$variant" == loop5x3 ]]; then
      for loops in 1 2 3; do bash scripts/loop/eval.sh "$checkpoint" "$loops"; done
    else
      bash scripts/loop/eval.sh "$checkpoint" 1
    fi
  done
done
"$python_bin" scripts/loop/summarize.py "$run_root" --output "$run_root/comparison.csv"

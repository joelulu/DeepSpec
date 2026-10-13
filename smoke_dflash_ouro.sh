#!/usr/bin/env bash
set -euo pipefail
repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$repo_root"
export PYTHONPATH="$repo_root${PYTHONPATH:+:$PYTHONPATH}"
source scripts/loop/settings.sh
export RUN_ROOT="${OURO_SMOKE_ROOT:-$repo_root/runs/dflash_ouro_smoke}"
export EPOCHS=1 GATE_EPOCHS=1 SMOKE_STEPS=2 LOCAL_BATCH=1 ANCHORS=2
export MIN_EXIT_LOOP=3 LOOP_TRANSFER=raw LOOP_LOSS=late
gpu_count=$("${PYTHON:-python}" -c 'import torch; print(torch.cuda.device_count())')
if (( gpu_count == 0 )); then
  echo "No visible CUDA GPU; set CUDA_VISIBLE_DEVICES and activate your DeepSpec environment" >&2
  exit 1
fi
export GLOBAL_BATCH=$((gpu_count * 2))
bash train_dflash_loop.sh baseline 1 5
baseline_root="$RUN_ROOT/baseline/deepspec"
baseline=$("${PYTHON:-python}" -c 'import pathlib,sys; paths=list(pathlib.Path(sys.argv[1]).glob("*/step_latest")); assert len(paths)==1, paths; print(paths[0])' "$baseline_root")
bash train_dflash_loop.sh A "$baseline"
bash train_dflash_loop.sh B 1 5
export EVAL_REPEATS=1 MAX_SAMPLES=2 MAX_NEW_TOKENS=32 EXIT_Q_LIST="0.5 0.8"
for stage in A_gate B_joint B_gate; do
  checkpoint=$("${PYTHON:-python}" -c 'import pathlib,sys; paths=list(pathlib.Path(sys.argv[1]).glob("*/step_latest")); assert len(paths)==1, paths; print(paths[0])' "$RUN_ROOT/$stage/deepspec")
  bash eval_dflash_loop.sh both "$checkpoint"
done

#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
source scripts/loop/settings.sh
export CUDA_VISIBLE_DEVICES=${MATRIX_GPUS:-0,1,2,3,4,5,6,7}
# Exercise new normalization, final-only loss, FSDP, held-out evaluation and save.
# This directory is separate from the three-epoch experiments.
export RUN_ROOT="$PWD/runs/dflash_flex_smoke"
export SMOKE_STEPS=2 LOOP_TRANSFER=rmsnorm LOOP_LOSS=final
bash train_dflash_loop.sh 5 3
percent=$("${PYTHON:-python}" -c 'from decimal import Decimal; import sys; print(format(Decimal(sys.argv[1]).normalize(), "f"))' "$DATA_PERCENT")
checkpoint="$RUN_ROOT/deepspec/dflash_L5_R3_p${percent}_e${EPOCHS}_rmsnorm_final_seed${SEED}/step_latest"
MAX_SAMPLES=2 MAX_NEW_TOKENS=32 bash eval_temp.sh "$checkpoint"

#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
source scripts/loop/settings.sh
export CUDA_VISIBLE_DEVICES=${MATRIX_GPUS:-0,1,2,3,4,5,6,7}
# All eight GPUs train one group at a time. A failure stops the queue.
export LOOP_LOSS=final
export LOOP_TRANSFER=raw
bash train_dflash_loop.sh 5 1
bash train_dflash_loop.sh 15 1
bash train_dflash_loop.sh 1 5
bash train_dflash_loop.sh 5 3
export LOOP_TRANSFER=rmsnorm
bash train_dflash_loop.sh 5 3

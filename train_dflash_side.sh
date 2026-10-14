#!/usr/bin/env bash
set -euo pipefail
repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$repo_root"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export RUN_ROOT=${RUN_ROOT:-$PWD/runs/dflash_side}
export DATA_PERCENT=${DATA_PERCENT:-1}
export EPOCHS=${EPOCHS:-1}
export LOCAL_BATCH=${LOCAL_BATCH:-2}
export GLOBAL_BATCH=${GLOBAL_BATCH:-64}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
source scripts/loop/settings.sh
exec "${PYTHON:-python}" scripts/loop/launch_side.py "$@"

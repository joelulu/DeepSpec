#!/usr/bin/env bash
set -euo pipefail
repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$repo_root"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo 'Usage: bash eval_dflash_side.sh CHECKPOINT [serial|overlap|both]' >&2
  exit 2
fi
export RUN_ROOT=${RUN_ROOT:-$PWD/runs/dflash_side}
source scripts/loop/settings.sh
exec "${PYTHON:-python}" scripts/loop/eval_side.py "$@"

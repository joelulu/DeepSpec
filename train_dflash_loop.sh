#!/usr/bin/env bash
set -euo pipefail
repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$repo_root"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
if [[ $# == 1 ]]; then
  case "$1" in
    5l) set -- 5 1 ;; 15l) set -- 15 1 ;; loop5x3) set -- 5 3 ;;
    *) echo "Usage: bash train_dflash_loop.sh LAYERS LOOPS" >&2; exit 2 ;;
  esac
fi
if [[ $# != 2 || ! $1 =~ ^[1-9][0-9]*$ || ! $2 =~ ^[1-9][0-9]*$ ]]; then
  echo "Usage: bash train_dflash_loop.sh LAYERS LOOPS (positive integers)" >&2
  exit 2
fi
source scripts/loop/settings.sh
"${PYTHON:-python}" scripts/loop/launch_flex.py "$1" "$2"

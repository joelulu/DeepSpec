#!/usr/bin/env bash
set -euo pipefail
repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$repo_root"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
source scripts/side/settings.sh
exec "${PYTHON:-python}" scripts/side/evaluate.py "$@" --algorithm dspark

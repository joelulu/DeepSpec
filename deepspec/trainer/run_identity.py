"""Reject accidental resume with a different experiment or data selection."""
import json
from pathlib import Path
from deepspec.data.target_cache_dataset import atomic_json_dump


def check_run_identity(root, identity, *, resuming=False, write=False):
    path = Path(root) / "run_identity.json"
    if path.exists():
        if json.loads(path.read_text()) != identity:
            raise ValueError("Experiment settings/data differ from this run. Use a new RUN_ROOT or EXP_NAME; do not resume it.")
    elif resuming:
        raise ValueError("Missing run_identity.json for flex resume. Use a new experiment directory.")
    elif write:
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_json_dump(identity, str(path))

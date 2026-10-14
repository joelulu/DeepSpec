"""Serial-request quality/throughput evaluation, with both branch schedules."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.loop.eval_ouro import build_evaluations


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("checkpoint")
    parser.add_argument("execution", choices=("serial", "overlap", "both"), nargs="?", default="both")
    args = parser.parse_args()
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=os.environ.get("EVAL_GPU", "0"))
    cfg = json.loads((Path(args.checkpoint) / "config.json").read_text())
    has_side = cfg.get("side_loop_mode", "none") != "none"
    if args.execution == "overlap" and not has_side:
        parser.error("baseline has no side branch to overlap")
    executions = ["serial", "overlap"] if args.execution == "both" and has_side else ["serial" if not has_side else args.execution]
    root, commands = build_evaluations("fixed", args.checkpoint, env)
    expanded = []
    for command in commands:
        for execution in executions:
            current = list(command)
            index = current.index("--output-json") + 1
            output = Path(current[index])
            current[index] = str(output.parent / execution / output.name)
            current += ["--side-execution", execution]
            expanded.append(current)
    print(json.dumps(dict(result_root=str(root), commands=expanded), indent=2), flush=True)
    if env.get("DRY_RUN") == "1":
        return
    for command in expanded:
        subprocess.run(command, env=env, check=True)
    subprocess.run([sys.executable, "scripts/loop/summarize.py", str(root),
                    "--output", str(root / "comparison.csv")], env=env, check=True)


if __name__ == "__main__":
    main()

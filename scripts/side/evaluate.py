"""Evaluate trained DFlash / DSpark side checkpoints on serial requests."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def build_evaluations(checkpoint, env, algorithm=None, execution="both"):
    checkpoint = Path(checkpoint).resolve()
    cfg = json.loads((checkpoint / "config.json").read_text())
    actual = "dspark" if int(cfg.get("markov_rank", 0)) > 0 else "dflash"
    if algorithm is not None and actual != algorithm:
        raise ValueError(f"Requested {algorithm}, but checkpoint is {actual}")
    if actual == "dflash" and cfg.get("enable_confidence_head", False):
        raise ValueError("DFlash evaluation requires a plain DFlash checkpoint")
    if actual == "dspark" and not cfg.get("enable_confidence_head", False):
        raise ValueError("DSpark evaluation requires its trained confidence head")
    if int(cfg.get("num_loops", 1)) != 1 or int(cfg["num_hidden_layers"]) != 5:
        raise ValueError("Side evaluation requires a five-layer, one-main-pass checkpoint")
    if execution not in ("serial", "overlap", "both"):
        raise ValueError("execution must be serial|overlap|both")
    has_side = cfg.get("side_loop_mode", "none") != "none"
    if execution == "overlap" and not has_side:
        raise ValueError("baseline has no side branch to overlap")
    executions = (["serial", "overlap"] if execution == "both" and has_side else
                  ["serial" if not has_side else execution])
    threshold = float(env.get("CONFIDENCE_THRESHOLD", "0"))
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("CONFIDENCE_THRESHOLD must be in [0, 1]")
    if actual == "dflash" and threshold != 0:
        raise ValueError("Confidence truncation requires a DSpark checkpoint")
    target = env.get("TARGET_MODEL")
    if not target:
        target = json.loads((Path(env["SOURCE_CACHE"]) / "manifest.json").read_text())["target_model_name_or_path"]
    key = checkpoint.parent.name + "_" + hashlib.sha256(str(checkpoint).encode()).hexdigest()[:8]
    root = Path(env.get("RESULT_ROOT", str(Path(env["RUN_ROOT"]) / "results" / key)))
    repeats = int(env.get("EVAL_REPEATS", "3"))
    if repeats < 1:
        raise ValueError("EVAL_REPEATS must be positive")
    commands = []
    for repeat in range(1, repeats + 1):
        for schedule in executions:
            output = root / f"repeat{repeat}" / schedule / "eval_loop1.json"
            commands.append([sys.executable, "eval.py", "--target_name_or_path", target,
                "--draft_name_or_path", str(checkpoint), "--tasks", env.get("TASKS", "gsm8k"),
                "--max-samples", env.get("MAX_SAMPLES", "128"),
                "--max-new-tokens", env.get("MAX_NEW_TOKENS", "256"),
                "--temperature", env.get("TEMPERATURE", "0"), "--seed", env.get("SEED", "42"),
                "--confidence-threshold", str(threshold), "--profile", "--num-loops", "1",
                "--warmup-samples", env.get("WARMUP_SAMPLES", "3"),
                "--output-json", str(output), "--side-execution", schedule])
    return root, commands


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("checkpoint")
    parser.add_argument("execution", choices=("serial", "overlap", "both"), nargs="?", default="both")
    parser.add_argument("--algorithm", choices=("dflash", "dspark"), default=None)
    args = parser.parse_args()
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=os.environ.get("EVAL_GPU", "0"))
    root, commands = build_evaluations(args.checkpoint, env, args.algorithm, args.execution)
    print(json.dumps(dict(algorithm=args.algorithm, result_root=str(root), commands=commands), indent=2), flush=True)
    if env.get("DRY_RUN") == "1":
        return
    for command in commands:
        subprocess.run(command, env=env, check=True)
    subprocess.run([sys.executable, "scripts/side/summarize.py", str(root),
                    "--output", str(root / "comparison.csv")], env=env, check=True)


if __name__ == "__main__":
    main()

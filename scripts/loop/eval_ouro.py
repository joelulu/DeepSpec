"""Evaluate every fixed depth and/or real block-level Q-exit for one checkpoint."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def build_evaluations(mode, checkpoint, env):
    checkpoint = Path(checkpoint).resolve()
    cfg = json.loads((checkpoint / "config.json").read_text())
    loops = int(cfg.get("num_loops", 1))
    if mode in ("gate", "both") and not (cfg.get("enable_exit_gate") and cfg.get("exit_gate_trained")):
        raise ValueError("Gate evaluation requires a checkpoint with a trained exit gate")
    target = env.get("TARGET_MODEL")
    if not target:
        target = json.loads((Path(env["SOURCE_CACHE"]) / "manifest.json").read_text())["target_model_name_or_path"]
    key = checkpoint.parent.name + "_" + hashlib.sha256(str(checkpoint).encode()).hexdigest()[:8]
    root = Path(env.get("RESULT_ROOT", str(Path(env["RUN_ROOT"]) / "results" / key)))
    requests = [(f"loop{r}", ["--num-loops", str(r)]) for r in range(1, loops + 1)] if mode in ("fixed", "both") else []
    if mode in ("gate", "both"):
        for value in env.get("EXIT_Q_LIST", "0.5 0.8 0.95").split():
            if not 0 < float(value) <= 1:
                raise ValueError("EXIT_Q_LIST values must be in (0, 1]")
            requests.append((f"loop_gate_q{value}", ["--exit-gate", "--exit-q", value]))
        if not env.get("EXIT_Q_LIST", "0.5 0.8 0.95").split():
            raise ValueError("EXIT_Q_LIST cannot be empty")
    repeats = int(env.get("EVAL_REPEATS", "1"))
    if repeats < 1:
        raise ValueError("EVAL_REPEATS must be positive")
    commands = []
    for repeat in range(1, repeats + 1):
        for name, options in requests:
            output = root / f"repeat{repeat}" / f"eval_{name}.json"
            command = [sys.executable, "eval.py", "--target_name_or_path", target,
                "--draft_name_or_path", str(checkpoint), "--tasks", env.get("TASKS", "gsm8k"),
                "--max-samples", env.get("MAX_SAMPLES", "16"),
                "--max-new-tokens", env.get("MAX_NEW_TOKENS", "256"),
                "--temperature", env.get("TEMPERATURE", "0"), "--seed", env.get("SEED", "42"),
                "--confidence-threshold", "0", "--profile",
                "--warmup-samples", env.get("WARMUP_SAMPLES", "1"), "--output-json", str(output)] + options
            commands.append(command)
    return root, commands


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("mode", choices=["fixed", "gate", "both"])
    parser.add_argument("checkpoint")
    args = parser.parse_args()
    env = dict(os.environ)
    # Existing evaluator runs serial requests per GPU. Use one GPU for comparisons.
    env["CUDA_VISIBLE_DEVICES"] = env.get("EVAL_GPU", "0")
    root, commands = build_evaluations(args.mode, args.checkpoint, env)
    print(json.dumps(dict(result_root=str(root), commands=commands), indent=2), flush=True)
    if env.get("DRY_RUN") == "1":
        return
    for command in commands:
        subprocess.run(command, env=env, check=True)
    subprocess.run([sys.executable, "scripts/loop/summarize.py", str(root),
                    "--output", str(root / "comparison.csv")], env=env, check=True)


if __name__ == "__main__":
    main()

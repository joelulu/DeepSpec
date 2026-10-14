"""Matched five-layer baseline, no-head fusion A, and grouped token-feedback B."""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.loop.launch_flex import build_launch


def build_side_launch(variant, env):
    if variant not in ("baseline", "A", "B"):
        raise ValueError("variant must be baseline|A|B")
    plan, _, opts = build_launch(5, 1, env)
    steps, group = int(env.get("SIDE_LOOPS", "5")), int(env.get("GROUP_SIZE", "3"))
    block = int(env.get("BLOCK_SIZE", str(steps * group + 1 if variant == "B" else 16)))
    rank = int(env.get("FUSION_RANK", "128"))
    if min(steps, group, block, rank) < 1:
        raise ValueError("Side loops, group, block and fusion rank must be positive")
    if block < 2:
        raise ValueError("BLOCK_SIZE must include an anchor plus at least one prediction")
    if variant == "B" and block - 1 != steps * group:
        raise ValueError("B requires BLOCK_SIZE-1=SIDE_LOOPS*GROUP_SIZE (default 16-1=5*3)")
    mode = {"baseline": "none", "A": "fusion", "B": "grouped"}[variant]
    name = (f"dflash_side_{variant}_maskonly_b{block}_r{steps}_g{group}_k{rank}"
            f"_p{plan['data_percent']}_e{plan['epochs']}_seed{env['SEED']}")
    opts.update({"exp_name": name, "model.block_size": block,
                 "model.sample_from_anchor": False,
                 "model.side_loop_mode": mode, "model.side_loop_steps": steps,
                 "model.side_group_size": group, "model.side_fusion_rank": rank,
                 "model.side_aux_loss_alpha": float(env.get("SIDE_AUX_ALPHA", "0.2")),
                 "model.loop_loss_weights": [1.0], "model.loop_loss_mode": "final",
                 "model.loop_boundary_norm": False,
                 "model.init_checkpoint": env.get("INIT_CHECKPOINT") or None,
                 "model.allow_block_size_transfer": bool(env.get("INIT_CHECKPOINT")),
                 "model.allow_prediction_convention_transfer": bool(env.get("INIT_CHECKPOINT")),
                 "logging.diagnostic_steps": 0})
    root = Path(plan["checkpoint_root"]).parents[1]
    plan.update(experiment=name, variant=variant, block_size=block,
                predicted_tokens=block - 1, sample_from_anchor=False,
                side_loop_mode=mode, side_loop_steps=steps, group_size=group,
                main_layers=5, shared_first_layer=True, side_execution_train="serial",
                checkpoint_root=str(root / "deepspec" / name),
                tensorboard=str(root / "tensorboard" / "deepspec" / name))
    command = [sys.executable, "train.py", "--config", "config/loop/dflash_flex_qwen3_4b.py"]
    for key, value in opts.items():
        command += ["--opts", key + "=" + json.dumps(value)]
    return plan, command, opts


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("variant", choices=("baseline", "A", "B"))
    args = parser.parse_args()
    plan, command, opts = build_side_launch(args.variant, os.environ)
    print(json.dumps(plan, indent=2), flush=True)
    if os.environ.get("DRY_RUN") == "1":
        print(shlex.join(command))
        return
    subprocess.run([sys.executable, "scripts/loop/preflight.py", "--cache", plan["source_cache"],
        "--target", opts["model.target_model_name_or_path"], "--global-batch", str(plan["global_batch"]),
        "--local-batch", str(plan["local_batch"]), "--data-percent", plan["data_percent"],
        "--holdout-samples", str(plan["holdout_samples"]), "--seed", str(opts["seed"]), "--check-cuda"], check=True)
    log_dir = Path(os.environ["RUN_ROOT"]) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / f"{plan['experiment']}.plan.json").write_text(json.dumps(plan, indent=2))
    with (log_dir / f"{plan['experiment']}.log").open("a", buffering=1) as handle:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, env=dict(os.environ, PYTHONUNBUFFERED="1"))
        for line in process.stdout:
            print(line, end="", flush=True)
            handle.write(line)
        code = process.wait()
    raise SystemExit(code)


if __name__ == "__main__":
    main()

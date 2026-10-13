"""Launch baseline, frozen gate training, or the two-stage Ouro adaptation."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from launch_flex import build_launch


def build_stage(stage, layers, loops, env, checkpoint=None, source_config=None):
    env = dict(env)
    if stage == "gate_only":
        if source_config is None:
            source_config = json.loads((Path(checkpoint) / "config.json").read_text())
        layers, loops = int(source_config["num_hidden_layers"]), int(source_config.get("num_loops", 1))
        env["LOOP_TRANSFER"] = "rmsnorm" if source_config.get("loop_boundary_norm", False) else "raw"
        env["EPOCHS"] = env.get("GATE_EPOCHS", "1")
        env["LR"] = env.get("GATE_LR", "1e-3")
    plan, _, opts = build_launch(layers, loops, env)
    label = "baseline" if stage == "baseline" else ("B_joint" if stage == "joint" else env.get("GATE_LABEL", "A_gate"))
    root = str(Path(env["RUN_ROOT"]) / label)
    opts.update({"run_root": root, "model.ouro_training_mode": stage,
                 "model.enable_exit_gate": stage != "baseline",
                 "model.min_exit_loop": int(env.get("MIN_EXIT_LOOP", "3")),
                 "model.ouro_beta": float(env.get("OURO_BETA", "0.1")),
                 "model.gate_gain_slope": float(env.get("GATE_GAIN_SLOPE", "50")),
                 "model.gate_gain_threshold": float(env.get("GATE_GAIN_THRESHOLD", "0.005")),
                 "model.init_checkpoint": str(Path(checkpoint).resolve()) if checkpoint else None,
                 "train.torch_compile": False,
                 "logging.save_training_state": env.get("SAVE_TRAINING_STATE", "1") == "1"})
    if stage == "gate_only":
        for key in ("block_size", "mask_token_id", "target_layer_ids", "loop_loss_weights"):
            if key in source_config:
                opts[f"model.{key}"] = source_config[key]
        # A trained gate's minimum is part of its semantics; B2 must preserve it.
        if source_config.get("enable_exit_gate", False):
            source_min = int(source_config["min_exit_loop"])
            if "MIN_EXIT_LOOP" in env and int(env["MIN_EXIT_LOOP"]) != source_min:
                raise ValueError("MIN_EXIT_LOOP must match the source gate")
            opts["model.min_exit_loop"] = source_min
    if stage != "baseline" and not 1 <= opts["model.min_exit_loop"] < loops:
        raise ValueError("MIN_EXIT_LOOP must be >= 1 and below the checkpoint's loop count")
    plan.update(stage=stage, source_checkpoint=opts["model.init_checkpoint"],
                checkpoint_root=f"{root}/deepspec/{plan['experiment']}",
                tensorboard=f"{root}/tensorboard/deepspec/{plan['experiment']}",
                minimum_exit=opts["model.min_exit_loop"] if stage != "baseline" else None)
    command = [sys.executable, "train.py", "--config", "config/loop/dflash_flex_qwen3_4b.py"]
    for key, value in opts.items():
        command += ["--opts", key + "=" + json.dumps(value)]
    return plan, command, opts


def execute_stage(stage, layers, loops, env, checkpoint=None, source_config=None):
    plan, command, opts = build_stage(stage, layers, loops, env, checkpoint, source_config)
    print(json.dumps(dict(plan=plan, overrides=opts, command=command), indent=2), flush=True)
    latest = Path(plan["checkpoint_root"]) / "step_latest"
    if env.get("DRY_RUN") == "1":
        return latest, opts
    subprocess.run([sys.executable, "scripts/loop/preflight.py", "--cache", plan["source_cache"],
        "--target", opts["model.target_model_name_or_path"], "--global-batch", str(plan["global_batch"]),
        "--local-batch", str(plan["local_batch"]), "--data-percent", plan["data_percent"],
        "--holdout-samples", str(plan["holdout_samples"]), "--seed", str(opts["seed"]), "--check-cuda"], check=True)
    log_dir = Path(env["RUN_ROOT"]) / "logs" / plan["stage"]
    log_dir.mkdir(parents=True, exist_ok=True)
    with (log_dir / f"{plan['experiment']}.log").open("a", buffering=1) as handle:
        handle.write(json.dumps(plan) + "\n")
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, env=dict(os.environ, PYTHONUNBUFFERED="1"))
        for line in process.stdout:
            sys.stdout.write(line); sys.stdout.flush(); handle.write(line)
        code = process.wait()
    if code:
        raise SystemExit(code)
    if not (latest / "config.json").is_file():
        raise RuntimeError(f"Training did not produce {latest}")
    print(f"Stage {stage} checkpoint: {latest}", flush=True)
    return latest, opts


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("mode", choices=["baseline", "A", "B", "B-gate"])
    parser.add_argument("values", nargs="+")
    args = parser.parse_args()
    env = dict(os.environ)
    if args.mode in ("A", "B-gate"):
        if len(args.values) != 1:
            parser.error("A/B-gate requires one source checkpoint")
        if args.mode == "B-gate":
            cfg = json.loads((Path(args.values[0]) / "config.json").read_text())
            if not cfg.get("enable_exit_gate") or not cfg.get("exit_gate_trained"):
                parser.error("B-gate requires a checkpoint with an already trained gate")
            env["GATE_LABEL"] = "B_gate"
        execute_stage("gate_only", None, None, env, args.values[0])
    else:
        if len(args.values) != 2:
            parser.error("baseline/B requires LAYERS LOOPS")
        layers, loops = map(int, args.values)
        if args.mode == "baseline":
            execute_stage("baseline", layers, loops, env)
        else:
            checkpoint, opts = execute_stage("joint", layers, loops, env)
            env["GATE_LABEL"] = "B_gate"
            # DRY_RUN can describe stage II before stage I exists.
            hypothetical = {key.removeprefix("model."): value for key, value in opts.items() if key.startswith("model.")}
            hypothetical["num_hidden_layers"] = layers
            execute_stage("gate_only", layers, loops, env, checkpoint,
                          hypothetical if env.get("DRY_RUN") == "1" else None)


if __name__ == "__main__":
    main()

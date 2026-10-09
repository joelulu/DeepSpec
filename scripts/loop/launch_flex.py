"""Resolve the logical data view, epoch schedule and config overrides before CUDA."""
import argparse
from decimal import Decimal
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from deepspec.data.cache_selection import select_cache_ids


def build_launch(layers, loops, env):
    if min(layers, loops) < 1:
        raise ValueError("Layers and loops must be positive")
    source = Path(env["SOURCE_CACHE"]).resolve()
    manifest = json.loads((source / "manifest.json").read_text())
    percent = format(Decimal(env["DATA_PERCENT"]).normalize(), "f")
    seed, held = int(env["SEED"]), int(env["HOLDOUT_SAMPLES"])
    ids = select_cache_ids(int(manifest["num_samples"]), percent, seed, held)
    count = int(manifest["num_samples"]) if ids is None else len(ids)
    epochs, local, batch = (int(env[k]) for k in ("EPOCHS", "LOCAL_BATCH", "GLOBAL_BATCH"))
    if min(epochs, local, batch) < 1:
        raise ValueError("Epochs and batch sizes must be positive")
    steps_per_epoch = count // batch
    if not steps_per_epoch:
        raise ValueError(f"Only {count} selected samples; need at least one global batch of {batch}")
    transfer, mode = env["LOOP_TRANSFER"], env["LOOP_LOSS"]
    if transfer not in ("raw", "rmsnorm") or mode not in ("final", "uniform", "late"):
        raise ValueError("LOOP_TRANSFER must be raw|rmsnorm; LOOP_LOSS must be final|uniform|late")
    weights = ([0.0] * (loops - 1) + [1.0]) if mode == "final" else ([1.0] * loops if mode == "uniform" else list(range(1, loops + 1)))
    if min(int(env["LOG_EVERY"]), int(env["DIAGNOSTIC_EVERY"])) < 0 or int(env["LOG_EVERY"]) == 0:
        raise ValueError("LOG_EVERY must be positive; DIAGNOSTIC_EVERY may be zero")
    name = f"dflash_L{layers}_R{loops}_p{percent}_e{epochs}_{transfer}_{mode}_seed{seed}"
    root = str(Path(env["RUN_ROOT"]).resolve())
    opts = {
        "model.target_model_name_or_path": manifest["target_model_name_or_path"],
        "model.target_layer_ids": manifest["target_layer_ids"],
        "model.num_draft_layers": layers, "model.num_loops": loops,
        "model.loop_loss_weights": weights, "model.loop_loss_mode": mode,
        "model.loop_boundary_norm": transfer == "rmsnorm", "model.num_anchors": int(env["ANCHORS"]),
        "data.target_cache_path": str(source), "data.data_percent": float(percent), "data.holdout_samples": held,
        "seed": seed, "run_root": root, "exp_name": name,
        "train.num_train_epochs": epochs, "train.max_train_steps": None,
        "train.local_batch_size": local, "train.global_batch_size": batch,
        "train.lr": float(env["LR"]), "train.sharding_strategy": env["SHARDING"],
        "logging.checkpointing_steps": steps_per_epoch, "logging.logging_steps": int(env["LOG_EVERY"]),
        "logging.diagnostic_steps": int(env["DIAGNOSTIC_EVERY"]),
    }
    smoke_steps = int(env.get("SMOKE_STEPS", "0"))
    if smoke_steps < 0 or smoke_steps > epochs * steps_per_epoch:
        raise ValueError("SMOKE_STEPS must be between 0 and the full training budget")
    if smoke_steps:
        opts["train.max_train_steps"] = smoke_steps
        opts["logging.checkpointing_steps"] = smoke_steps
    command = [sys.executable, "train.py", "--config", "config/loop/dflash_flex_qwen3_4b.py"]
    for key, value in opts.items():
        command += ["--opts", key + "=" + json.dumps(value)]
    plan = dict(experiment=name, source_cache=str(source), data_percent=percent, train_samples=count,
        holdout_samples=held, visible_gpus=env.get("CUDA_VISIBLE_DEVICES", ""), layers=layers, loops=loops, transfer=transfer, loss_weights=weights,
        local_batch=local, global_batch=batch, epochs=epochs, steps_per_epoch=steps_per_epoch,
        max_train_steps=smoke_steps or epochs * steps_per_epoch, checkpoint_every=smoke_steps or steps_per_epoch,
        smoke=bool(smoke_steps),
        checkpoint_root=f"{root}/deepspec/{name}", tensorboard=f"{root}/tensorboard/deepspec/{name}")
    return plan, command, opts


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("layers", type=int); parser.add_argument("loops", type=int)
    args = parser.parse_args()
    plan, command, opts = build_launch(args.layers, args.loops, os.environ)
    print(json.dumps(plan, indent=2), flush=True)
    if os.environ.get("DRY_RUN") == "1":
        print(shlex.join(command)); return
    subprocess.run([sys.executable, "scripts/loop/preflight.py", "--cache", plan["source_cache"],
        "--target", opts["model.target_model_name_or_path"], "--global-batch", str(plan["global_batch"]),
        "--local-batch", str(plan["local_batch"]), "--data-percent", plan["data_percent"],
        "--holdout-samples", str(plan["holdout_samples"]), "--seed", str(opts["seed"]), "--check-cuda"], check=True)
    # A tee-like console log without shell interpolation; propagate train failures.
    log_dir = Path(os.environ["RUN_ROOT"]) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    with (log_dir / f"{plan['experiment']}.log").open("a", buffering=1) as handle:
        handle.write(json.dumps(plan) + "\n")
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, env=dict(os.environ, PYTHONUNBUFFERED="1"))
        for line in process.stdout:
            sys.stdout.write(line); sys.stdout.flush(); handle.write(line)
        code = process.wait()
    raise SystemExit(code)


if __name__ == "__main__":
    main()

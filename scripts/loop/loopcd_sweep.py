"""Compare frozen-checkpoint LoopCD with the original readout on one GPU."""
import argparse
import csv
import itertools
import json
import math
import os
from pathlib import Path
import random
import statistics
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--target", required=True)
    parser.add_argument("--draft", required=True)
    parser.add_argument("--num-loops", type=int, default=5)
    parser.add_argument("--early-loops", type=int, nargs="+", default=[1])
    parser.add_argument("--lambdas", type=float, nargs="+", default=[0.1, 0.2, 0.3])
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--tasks", default="gsm8k")
    parser.add_argument("--max-samples", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup-samples", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if any(not 1 <= early < args.num_loops for early in args.early_loops):
        parser.error("each early loop must satisfy 1 <= early < num-loops")
    if any(not math.isfinite(value) or value < 0 for value in args.lambdas):
        parser.error("lambdas must be finite and nonnegative")
    if not math.isfinite(args.alpha) or not 0 < args.alpha <= 1:
        parser.error("alpha must be in (0, 1]")
    if min(args.repeats, args.max_samples, args.max_new_tokens) < 1 or args.warmup_samples < 0:
        parser.error("repeats, samples and tokens must be positive; warmup nonnegative")
    if not math.isfinite(args.temperature) or args.temperature < 0:
        parser.error("temperature must be finite and nonnegative")
    args.output_dir = args.output_dir.resolve()
    return args


def build_runs(args):
    variants = [("baseline", None, None)]
    variants += [(f"early{early}_lambda{strength:g}", early, strength)
                 for early, strength in itertools.product(
                     sorted(set(args.early_loops)), sorted(set(args.lambdas)))]
    # Keep dataset/sampling seeds fixed, shuffle treatment order to limit drift.
    order_rng = random.Random(args.seed)
    runs = []
    for repeat in range(1, args.repeats + 1):
        order = list(variants)
        order_rng.shuffle(order)
        for name, early, strength in order:
            output = args.output_dir / f"repeat{repeat}" / name / f"eval_loop{args.num_loops}.json"
            command = [sys.executable, str(REPO_ROOT / "eval.py"),
                       "--target_name_or_path", args.target, "--draft_name_or_path", args.draft,
                       "--num-loops", str(args.num_loops), "--tasks", args.tasks,
                       "--max-samples", str(args.max_samples), "--max-new-tokens", str(args.max_new_tokens),
                       "--temperature", str(args.temperature), "--seed", str(args.seed),
                       "--confidence-threshold", "0", "--profile",
                       "--warmup-samples", str(args.warmup_samples), "--output-json", str(output)]
            if early is not None:
                command += ["--loopcd", "--loopcd-early-loop", str(early),
                            "--loopcd-lambda", str(strength), "--loopcd-alpha", str(args.alpha)]
            runs.append(dict(repeat=repeat, variant=name, command=command, output=str(output)))
    return runs


def summarize(runs, output_dir):
    rows = []
    baselines = {}
    for run in runs:
        payload = json.loads(Path(run["output"]).read_text(encoding="utf-8"))
        for metric in payload["metrics"]:
            # This repository's MAT includes the target bonus token. Conditional
            # rates telescope to accepted_at_pos / proposal_count, including EOS
            # rounds where position proposal denominators can differ.
            survival, accepted_draft = 1.0, 0.0
            for rate in metric["conditional_accept_rates_by_position"]:
                survival *= rate or 0.0
                accepted_draft += survival
            row = dict(repeat=run["repeat"], variant=run["variant"], dataset=metric["dataset"],
                       samples=metric["num_samples"], accept_length=metric["acceptance_length"],
                       accepted_draft_per_round=accepted_draft,
                       serial_decode_tps=metric["serial_decode_tokens_per_second"],
                       serial_generation_tps=metric["serial_generation_tokens_per_second"],
                       draft_ms_per_round=metric["draft_ms"] * metric["round_wall_ms"] / metric["decode_wall_ms"],
                       conditional_accept_rates=json.dumps(metric["conditional_accept_rates_by_position"]))
            rows.append(row)
            if run["variant"] == "baseline":
                baselines[(run["repeat"], metric["dataset"])] = row
    if not rows:
        raise ValueError("No evaluation metrics found")
    for row in rows:
        baseline = baselines[(row["repeat"], row["dataset"])]
        row["accept_length_delta"] = row["accept_length"] - baseline["accept_length"]
        row["accepted_draft_delta"] = row["accepted_draft_per_round"] - baseline["accepted_draft_per_round"]
        row["decode_tps_gain_pct"] = 100 * (row["serial_decode_tps"] / baseline["serial_decode_tps"] - 1)
        row["generation_tps_gain_pct"] = 100 * (row["serial_generation_tps"] / baseline["serial_generation_tps"] - 1)
    write_csv(output_dir / "comparison.csv", rows)
    medians = []
    for variant, dataset in sorted({(row["variant"], row["dataset"]) for row in rows}):
        group = [row for row in rows if (row["variant"], row["dataset"]) == (variant, dataset)]
        median = dict(variant=variant, dataset=dataset, repeats=len(group), samples=group[0]["samples"])
        for key in ("accept_length", "accepted_draft_per_round", "serial_decode_tps",
                    "serial_generation_tps", "draft_ms_per_round", "accept_length_delta",
                    "accepted_draft_delta", "decode_tps_gain_pct", "generation_tps_gain_pct"):
            median[key] = statistics.median(row[key] for row in group)
        medians.append(median)
    write_csv(output_dir / "summary.csv", medians)
    return medians


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None):
    args = parse_args(argv)
    runs = build_runs(args)
    if args.dry_run:
        print(json.dumps(runs, indent=2))
        return
    import torch
    if torch.cuda.device_count() != 1:
        raise SystemExit("Expose exactly one GPU with CUDA_VISIBLE_DEVICES for serial timing")
    # Refuse to mix stale outputs with a new experiment.
    args.output_dir.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    manifest = dict(args=vars(args) | {"output_dir": str(args.output_dir)}, runs=runs,
                    device=torch.cuda.get_device_name(0),
                    git_sha=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip(),
                    timing_scope="serial requests on one GPU; includes LoopCD readout overhead")
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    for run in runs:
        output = Path(run["output"])
        output.parent.mkdir(parents=True, exist_ok=True)
        print(f"Repeat {run['repeat']}: {run['variant']}", flush=True)
        with output.with_suffix(".log").open("w", encoding="utf-8") as log:
            subprocess.run(run["command"], cwd=REPO_ROOT, env=env, stdout=log,
                           stderr=subprocess.STDOUT, check=True)
        if not output.is_file():
            raise RuntimeError(f"Evaluator produced no JSON; inspect {output.with_suffix('.log')}")
    print(json.dumps(summarize(runs, args.output_dir), indent=2))
    print(f"Results: {args.output_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()

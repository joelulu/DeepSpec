"""Summarize per-checkpoint eval JSON without equating MAT with speedup."""
import argparse
import csv
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    seen = set()
    for path in sorted(args.root.rglob("eval_loop*.json")):
        # step_latest is a symlink; do not double count its checkpoint.
        if path.resolve() in seen:
            continue
        seen.add(path.resolve())
        payload = json.loads(path.read_text())
        for metric in payload["metrics"]:
            rows.append(dict(checkpoint=payload["args"]["draft_name_or_path"],
                target=payload["args"]["target_name_or_path"],
                temperature=payload["args"]["temperature"],
                max_new_tokens=payload["args"]["max_new_tokens"],
                seed=payload["args"]["seed"],
                num_layers=payload["num_draft_layers"], num_loops=payload["num_loops"],
                executed_depth=payload["num_draft_layers"] * payload["num_loops"],
                dataset=metric["dataset"], samples=metric["num_samples"],
                accept_length=metric["acceptance_length"],
                serial_decode_tps=metric.get("serial_decode_tokens_per_second", ""),
                round_wall_ms=metric.get("round_wall_ms", ""),
                draft_ms=metric.get("draft_ms", ""), verify_ms=metric.get("verify_ms", ""),
                conditional_accept_rates=json.dumps(metric["conditional_accept_rates_by_position"])))
    if not rows:
        raise SystemExit("No eval_loop*.json found under the requested root")
    def comparison_key(row):
        return tuple(row[key] for key in ("target", "temperature", "max_new_tokens", "seed", "dataset", "samples"))
    ar_rates = {comparison_key(row): float(row["serial_decode_tps"]) for row in rows
                if row["num_layers"] == 0 and row["serial_decode_tps"]}
    for row in rows:
        baseline = ar_rates.get(comparison_key(row))
        row["speedup_vs_ar"] = float(row["serial_decode_tps"]) / baseline if baseline and row["serial_decode_tps"] else ""
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()

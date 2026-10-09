"""Summarize depth heterogeneity and an offline fractional-utility oracle."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path


def validate_depths(rounds):
    if not rounds or not rounds[0]["exits"]:
        raise ValueError("Need nonempty rounds with exits")
    count = len(rounds[0]["exits"])
    for row in rounds:
        if len(row["exits"]) != count or any(e["num_loops"] != i + 1 or e["round_wall_ms"] <= 0
                for i, e in enumerate(row["exits"])):
            raise ValueError("Expected consistent ordered exits 1..N with positive timings")
    return count


def aggregate_oracle(rounds):
    # Dinkelbach: maximize sum(progress) / sum(time), not average per-block TPS.
    count = validate_depths(rounds)
    rate = max(sum(r["exits"][i]["progress_tokens"] for r in rounds) /
               sum(r["exits"][i]["round_wall_ms"] for r in rounds) for i in range(count))
    for _ in range(100):
        choices = [max(r["exits"], key=lambda e: (
            e["progress_tokens"] - rate * e["round_wall_ms"], -e["num_loops"])) for r in rounds]
        new_rate = sum(e["progress_tokens"] for e in choices) / sum(e["round_wall_ms"] for e in choices)
        if abs(new_rate - rate) < 1e-10:
            break
        rate = new_rate
    return choices, new_rate * 1000


def summarize(rounds):
    valid = [r for r in rounds if r["valid_for_utility"]]
    result = dict(rounds=len(rounds), utility_rounds=len(valid), excluded_terminal_or_budget_rounds=len(rounds) - len(valid))
    if not valid:
        return result
    count = validate_depths(valid)
    fixed = []
    for i in range(count):
        exits = [r["exits"][i] for r in valid]
        if any(e["num_loops"] != i + 1 or e["round_wall_ms"] <= 0 for e in exits):
            raise ValueError("Expected ordered exits 1..N with positive timings")
        fixed.append(dict(num_loops=i + 1,
            mean_accepted_draft_tokens=sum(e["accepted_draft_tokens"] for e in exits) / len(exits),
            mean_round_wall_ms=sum(e["round_wall_ms"] for e in exits) / len(exits),
            progress_per_second_proxy=sum(e["progress_tokens"] for e in exits) * 1000 / sum(e["round_wall_ms"] for e in exits)))
    choices, oracle_rate = aggregate_oracle(valid)
    best = max(fixed, key=lambda e: e["progress_per_second_proxy"])
    transitions = {}
    for i in range(count - 1):
        deltas = [r["exits"][i + 1]["accepted_draft_tokens"] - r["exits"][i]["accepted_draft_tokens"] for r in valid]
        transitions[f"{i+1}_to_{i+2}"] = dict(improved=sum(d > 0 for d in deltas),
            unchanged=sum(d == 0 for d in deltas), worsened=sum(d < 0 for d in deltas))
    return result | dict(fixed_depths=fixed, best_fixed_loop=best["num_loops"],
        per_block_best_loop_counts=dict(Counter(r["best_loop_by_proxy"] for r in valid)),
        aggregate_oracle_loop_counts=dict(Counter(e["num_loops"] for e in choices)),
        aggregate_oracle_progress_per_second_proxy=oracle_rate,
        oracle_proxy_ratio_vs_best_fixed=oracle_rate / best["progress_per_second_proxy"],
        acceptance_transitions=transitions)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    groups = defaultdict(list)
    with args.trace.open() as handle:
        for line in handle:
            row = json.loads(line)
            groups[row["dataset"]].append(row)
    result = dict(interpretation="Offline greedy utility proxy on fixed maximum-loop prefixes; measured timing noise and future information can inflate oracle gains. Excludes gate overhead; not real dynamic throughput.",
        datasets={name: summarize(rows) for name, rows in groups.items()})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

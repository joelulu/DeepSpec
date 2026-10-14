"""CUDA same-input timing of 5-layer main, layer-1 recurrence, serial and overlap.

Synthetic target features isolate execution cost, not quality or serving TPS.
Each trial includes context projection and the output head. Inner event timings
separate the shared first layer, main tail, side path (including its group heads)
and fusion. There is NO synchronization between the two overlapping branches.
"""
import argparse
import json
from pathlib import Path
import random
import statistics
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch
from transformers import AutoConfig

from deepspec.modeling.dspark.qwen3.modeling import Qwen3DSparkModel
from deepspec.modeling.dspark.side_loop import PathTimer, SIDE_CONFIG_DEFAULTS, prepare_paths, run_paths


def stats(values):
    ordered = sorted(values)
    return dict(mean=statistics.mean(values), median=statistics.median(values),
                p90=ordered[min(len(values) - 1, int(0.9 * len(values)))])


def load_model(args, device):
    config = AutoConfig.from_pretrained(args.draft)
    if int(config.num_hidden_layers) != 5 or int(getattr(config, "num_loops", 1)) != 1:
        raise ValueError("Timing requires a five-layer, one-main-pass checkpoint")
    source_mode = getattr(config, "side_loop_mode", "none")
    for key, default in SIDE_CONFIG_DEFAULTS.items():
        if not hasattr(config, key):
            setattr(config, key, default)
    if args.variant:
        config.side_loop_mode = {"A": "fusion", "B": "grouped"}[args.variant]
    if config.side_loop_mode == "none":
        raise ValueError("Use an A/B checkpoint, or --variant A|B for timing-only baseline adaptation")
    if args.side_loops is not None:
        config.side_loop_steps = args.side_loops
    if args.group_size is not None:
        config.side_group_size = args.group_size
    if config.side_loop_mode == "grouped":
        config.block_size = config.side_loop_steps * config.side_group_size
    if args.block_size is not None:
        config.block_size = args.block_size
    config.num_loops = 1
    config.loop_loss_weights = [1.0]
    model, info = Qwen3DSparkModel.from_pretrained(args.draft, config=config,
        dtype=torch.bfloat16, attn_implementation="sdpa", output_loading_info=True)
    allowed = {"side_fusion.down.weight", "side_fusion.up.weight"} if source_mode == "none" else set()
    if set(info["missing_keys"]) - allowed or info["unexpected_keys"] or info.get("mismatched_keys"):
        raise ValueError(f"Checkpoint loading mismatch: {info}")
    return model.to(device).eval(), source_mode == "none" or args.variant is not None


def summarize_rows(rows):
    result = {}
    for execution in ("main", "side", "serial", "overlap"):
        selected = [row for row in rows if row["execution"] == execution]
        result[execution] = {name: stats([row[name] for row in selected])
                             for name in selected[0] if name != "execution"}
    return result


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--draft", required=True)
    parser.add_argument("--variant", choices=("A", "B"), default=None)
    parser.add_argument("--side-loops", type=int)
    parser.add_argument("--group-size", type=int)
    parser.add_argument("--block-size", type=int)
    parser.add_argument("--contexts", default="1024,4096,16384")
    parser.add_argument("--batches", default="1")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    contexts, batches = ([int(v) for v in value.split(",")] for value in (args.contexts, args.batches))
    if args.warmup < 1 or args.repeats < 1 or min(contexts + batches) < 1:
        parser.error("contexts, batches, warmup and repeats must be positive")
    if not torch.cuda.is_available():
        parser.error("CUDA is required; CPU timings cannot answer overlap on H100")
    device = torch.device("cuda:0")
    torch.manual_seed(args.seed)
    model, timing_adapted = load_model(args, device)
    payload = dict(checkpoint=args.draft, device=torch.cuda.get_device_name(device),
        torch_version=torch.__version__, side_loop_mode=model.side_loop_mode,
        side_loop_steps=model.side_loop_steps, group_size=model.side_group_size,
        block_size=model.block_size, main_layers=5, shared_first_layer=True,
        timing_only_adaptation=timing_adapted, args={**vars(args), "output": str(args.output)},
        scope="synthetic full-context batch, SDPA, no target verification; not quality or serving throughput",
        results=[])
    rng = random.Random(args.seed)
    with torch.inference_mode():
        for context_length in contexts:
            for batch in batches:
                block = model.block_size
                features = torch.randn(batch, context_length,
                    len(model.target_layer_ids) * model.config.hidden_size, device=device, dtype=torch.bfloat16)
                ids = torch.full((batch, block), model.mask_token_id, device=device, dtype=torch.long)
                ids[:, 0] = 1
                noise = model.embed_tokens(ids)
                positions = torch.arange(context_length + block, device=device)[None].expand(batch, -1)

                def trial(execution, profile):
                    torch.cuda.synchronize(device)
                    begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    host_begin = time.perf_counter()
                    begin.record()
                    timer = PathTimer(profile, device)
                    prepared = prepare_paths(model, position_ids=positions, noise_embedding=noise,
                        target_hidden_states=features, timer=timer, is_causal=False)
                    hidden, _ = run_paths(model, prepared, execution=execution)
                    with timer.stage("output_head"):
                        tokens = model.compute_logits(hidden).argmax(dim=-1)
                    end.record()
                    torch.cuda.synchronize(device)
                    row = dict(execution=execution, gpu_total_ms=begin.elapsed_time(end),
                               wall_total_ms=(time.perf_counter() - host_begin) * 1000)
                    row.update(timer.resolve())
                    if execution != "side":
                        row["main_stack_5l_ms"] = row.get("shared_layer1_ms", 0) + row.get("main_tail_ms", 0)
                    if execution != "main":
                        row["side_loopN_ms"] = row.get("shared_layer1_ms", 0) + row.get("side_after_fork_ms", 0)
                    return row, tokens

                modes = ["main", "side", "serial", "overlap"]
                for _ in range(args.warmup):
                    for execution in modes:
                        trial(execution, False)
                # Fail loudly if overlapping the shared layer changes results.
                _, serial_ids = trial("serial", False)
                _, overlap_ids = trial("overlap", False)
                if not torch.equal(serial_ids, overlap_ids):
                    raise RuntimeError("Serial and overlapping schedules produced different tokens")
                # Uninstrumented latency measures the actual scheduling cost;
                # a separate pass adds inner CUDA events for component diagnosis.
                raw, components = [], []
                for _ in range(args.repeats):
                    order = list(modes)
                    rng.shuffle(order)
                    for execution in order:
                        row, _ = trial(execution, False)
                        raw.append({key: row[key] for key in ("execution", "gpu_total_ms", "wall_total_ms")})
                for _ in range(args.repeats):
                    order = list(modes)
                    rng.shuffle(order)
                    for execution in order:
                        row, _ = trial(execution, True)
                        components.append(row)
                summary = summarize_rows(raw)
                main_ms = summary["main"]["wall_total_ms"]["median"]
                serial_ms = summary["serial"]["wall_total_ms"]["median"]
                overlap_ms = summary["overlap"]["wall_total_ms"]["median"]
                result = dict(context_tokens=context_length, batch=batch,
                    latency=summary, component_timings=summarize_rows(components),
                    overlap_overhead_vs_main_pct=100 * (overlap_ms / main_ms - 1),
                    serial_overhead_vs_main_pct=100 * (serial_ms / main_ms - 1),
                    overlap_saving_vs_serial_pct=100 * (1 - overlap_ms / serial_ms),
                    serial_overlap_tokens_equal=True, raw_latency=raw, raw_components=components)
                payload["results"].append(result)
                print(json.dumps({key: value for key, value in result.items()
                                  if key not in ("raw_latency", "raw_components")}), flush=True)
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

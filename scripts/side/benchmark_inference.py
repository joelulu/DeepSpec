"""Time eight matched draft forwards inside real, single-request decoding."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import subprocess
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--target", required=True)
    parser.add_argument("--draft", required=True)
    parser.add_argument("--task", default="gsm8k")
    parser.add_argument("--dataset-root", default="eval_datasets")
    parser.add_argument("--max-samples", type=int, default=128,
                        help="Safety cap only; stop as soon as the draft-call budget is met")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--side-loops", type=int, default=5)
    parser.add_argument("--warmup-calls", type=int, default=5)
    parser.add_argument("--measure-calls", type=int, default=50)
    parser.add_argument("--context-buckets", default="1024,4096",
        help="Compatible context limit/trace ranges; timing uses exact shapes without padding")
    parser.add_argument("--modes", default="main_only,side_only,dual_stream,fusion_only")
    parser.add_argument("--backends", default="eager,cudagraph")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--profile-trace", action="store_true")
    parser.add_argument("--no-component-timings", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if min(args.max_samples, args.max_new_tokens, args.side_loops,
           args.warmup_calls, args.measure_calls) < 1:
        parser.error("sample/token/call budgets and side-loops must be positive")
    if args.block_size != 16:
        parser.error("This experiment fixes block-size=16")
    try:
        args.buckets = tuple(int(x) for x in args.context_buckets.split(","))
    except ValueError:
        parser.error("context-buckets must be comma-separated integers")
    if not args.buckets or min(args.buckets) < 1 or tuple(sorted(set(args.buckets))) != args.buckets:
        parser.error("context-buckets must be positive, unique and increasing")
    args.modes = tuple(args.modes.split(","))
    args.backends = tuple(args.backends.split(","))
    valid_modes = {"main_only", "side_only", "dual_stream", "fusion_only"}
    if (not args.modes or set(args.modes) - valid_modes or "main_only" not in args.modes
            or len(set(args.modes)) != len(args.modes)):
        parser.error("Choose unique timing modes and include main_only")
    if (not args.backends or set(args.backends) - {"eager", "cudagraph"}
            or len(set(args.backends)) != len(args.backends)):
        parser.error("Choose unique backends from eager,cudagraph")
    return args


def run(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from deepspec.data.parser import encode_chat_messages
    from deepspec.eval.base_evaluator import (assert_no_final_target_layer,
        generate_decoding_sample, load_and_process_dataset, resolve_stop_token_ids)
    from deepspec.eval.dspark.draft_ops import build_dspark_proposal, forward_dspark_draft_block
    from deepspec.eval.dspark.draft_timing import DraftTimingObserver, TimingComplete, validate_timing_model
    from deepspec.modeling.dspark.common import extract_context_feature
    from deepspec.modeling.dspark.context_cache import ContextKVCache
    from deepspec.modeling.dspark.qwen3 import Qwen3DSparkModel
    from deepspec.utils import seed_all

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError("Expose exactly one CUDA GPU, e.g. CUDA_VISIBLE_DEVICES=0")
    torch.cuda.set_device(0)
    device = torch.device("cuda:0")
    seed_all(args.seed)
    # Fail before loading weights if an existing result directory would be overwritten.
    if any((args.output / name).exists() for name in ("calls.jsonl", "summary.csv", "metadata.json")):
        raise ValueError("Output already contains results; choose a new --output")
    draft = Qwen3DSparkModel.from_pretrained(args.draft,
        dtype=torch.bfloat16, attn_implementation="sdpa").to(device).eval()
    draft.side_loop_steps = args.side_loops  # cost-only sweep; no quality claim
    validate_timing_model(draft)
    target = AutoModelForCausalLM.from_pretrained(args.target,
        dtype=torch.bfloat16, attn_implementation="sdpa").to(device).eval()
    assert_no_final_target_layer(target, draft.target_layer_ids)
    tokenizer = AutoTokenizer.from_pretrained(args.target)
    observer = DraftTimingObserver(draft, args.output,
        warmup_calls=args.warmup_calls, measure_calls=args.measure_calls,
        buckets=args.buckets, modes=args.modes, backends=args.backends,
        seed=args.seed, profile_trace=args.profile_trace,
        component_timings=not args.no_component_timings)
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = "unknown"
    observer.metadata.update(checkpoint=str(Path(args.draft).resolve()), target=args.target,
        task=args.task, git_revision=revision)
    draft.side_execution = "main_only"
    draft.draft_timing_observer = observer

    def init_context(initial_output, **kwargs):
        return SimpleNamespace(past_key_values_draft=ContextKVCache(),
            target_hidden_states=extract_context_feature(initial_output.hidden_states, draft.target_layer_ids))

    def propose(context, output_ids, position_ids, start, **kwargs):
        ids = torch.full((1, draft.block_size), draft.mask_token_id, device=device, dtype=torch.long)
        ids[:, 0] = output_ids[:, start]
        hidden = forward_dspark_draft_block(draft, draft_input_ids=ids,
            position_ids=position_ids, past_key_values_draft=context.past_key_values_draft,
            target_hidden_states=context.target_hidden_states, start=start, block_size=draft.block_size)
        return build_dspark_proposal(draft, draft_input_ids=ids, block_hidden=hidden,
            block_size=draft.block_size, temperature=0.0, confidence_threshold=0.0)

    def update(context, verification):
        context.target_hidden_states = extract_context_feature(
            verification.target_output.hidden_states, draft.target_layer_ids
        )[:, :verification.accepted_draft_tokens + 1]

    try:
        dataset = load_and_process_dataset(args.task, args.dataset_root)
        random.Random(args.seed).shuffle(dataset)
        stop_ids = resolve_stop_token_ids(target, tokenizer)
        for index, instance in enumerate(dataset[:args.max_samples]):
            observer.metadata["reference_samples_started"] = index + 1
            ids = encode_chat_messages(tokenizer,
                [{"role": "user", "content": instance["turns"][0]}],
                add_generation_prompt=True, enable_thinking=False).to(device)
            generate_decoding_sample(target_model=target, input_ids=ids,
                max_new_tokens=args.max_new_tokens, max_proposal_tokens=15,
                temperature=0.0, stop_token_ids=stop_ids, init_context=init_context,
                propose=propose, update=update, profile=False)
        if not observer.done:
            raise RuntimeError(f"Insufficient real draft calls: {observer.seen}; increase "
                "--max-new-tokens or --max-samples. Partial results have been retained.")
    except TimingComplete:
        pass
    except BaseException as exc:
        observer.metadata["error"] = str(exc)
        observer.flush(status="failed")
        raise
    finally:
        draft.draft_timing_observer = None
    summaries = observer.flush(status="complete")
    print("\nmode          backend     calls   wall_ms/call   GPU_ms/call   overhead")
    for row in summaries:
        print(f"{row['mode']:13s} {row['backend']:10s} {row['calls']:5d} "
              f"{row['wall_ms_mean']:14.4f} {row['gpu_ms_mean']:13.4f} "
              f"{row['overhead_wall_pct']:+9.2f}%")
    print(f"\nResults: {args.output / 'summary.csv'}", flush=True)


if __name__ == "__main__":
    run(parse_args())

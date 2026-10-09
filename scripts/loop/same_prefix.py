"""Collect greedy, same-prefix 1/2/3-loop counterfactuals on a 3-loop rollout."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
from deepspec.data.parser import encode_chat_messages
from deepspec.eval.base_evaluator import (assert_no_final_target_layer,
    has_stop_token, load_and_process_dataset, resolve_stop_token_ids)
from deepspec.eval.dspark.depth_probe import probe_depths
from deepspec.modeling.dspark.common import extract_context_feature
from deepspec.modeling.dspark.context_cache import ContextKVCache
from deepspec.modeling.dspark.qwen3 import Qwen3DSparkModel
from deepspec.utils import seed_all


@torch.inference_mode()
def scan_sample(target, draft, input_ids, stop_ids, args, dataset_name, sample_id, handle):
    device = input_ids.device
    prompt_length = input_ids.shape[1]
    max_length = prompt_length + args.max_new_tokens
    position_ids = torch.arange(max_length + draft.block_size + 1, device=device)[None]
    output_ids = torch.empty_like(position_ids)
    output_ids[:, :prompt_length] = input_ids
    target_cache, draft_cache = DynamicCache(), ContextKVCache()
    initial = target(input_ids=input_ids, past_key_values=target_cache, use_cache=True,
                     output_hidden_states=True, logits_to_keep=1)
    output_ids[:, prompt_length] = initial.logits[:, -1].argmax(-1)
    if has_stop_token(output_ids[:, prompt_length:prompt_length + 1], stop_ids):
        return 0
    hidden = extract_context_feature(initial.hidden_states, draft.target_layer_ids)
    del initial
    start, round_id = prompt_length, 0
    while start + 1 < max_length and round_id < args.max_rounds:
        prefix_hash = hashlib.sha256(output_ids[:, :start + 1].cpu().numpy().tobytes()).hexdigest()
        exits, selected = probe_depths(target=target, draft=draft, target_cache=target_cache,
            draft_cache=draft_cache, target_hidden_states=hidden, output_ids=output_ids,
            position_ids=position_ids, start=start, stop_token_ids=stop_ids, repeats=args.repeats)
        remaining = max_length - start - 1
        # Exclude all exits at EOS/budget boundaries from oracle utility labels.
        valid = all(not e["terminated"] and e["progress_tokens"] <= remaining for e in exits)
        best = max(exits, key=lambda e: (e["progress_per_second_proxy"], -e["num_loops"]))
        if handle is not None:
            handle.write(json.dumps(dict(dataset=dataset_name, sample_id=sample_id,
                round_id=round_id, context_length=start, prefix_sha256=prefix_hash,
                rollout_num_loops=draft.num_loops, valid_for_utility=valid,
                best_loop_by_proxy=best["num_loops"], exits=exits)) + "\n")
            handle.flush()
        committed = selected["committed_tokens"][:, :remaining]
        output_ids[:, start + 1:start + 1 + committed.shape[1]] = committed
        start += committed.shape[1]
        hidden, draft_cache = selected["target_hidden_states"], selected["draft_cache"]
        round_id += 1
        if selected["terminated"] or start + 1 >= max_length:
            break
    return round_id


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--target", required=True)
    parser.add_argument("--draft", required=True)
    parser.add_argument("--tasks", default="gsm8k,mbpp,alpaca")
    parser.add_argument("--dataset-root", default="eval_datasets")
    parser.add_argument("--max-samples", type=int, default=16)
    parser.add_argument("--max-rounds", type=int, default=40)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup-samples", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.max_samples, args.max_rounds, args.max_new_tokens, args.repeats) < 1 or args.warmup_samples < 0:
        parser.error("Counts must be positive; warmup must be nonnegative")
    if not torch.cuda.is_available():
        parser.error("CUDA is required for meaningful hardware timing")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    seed_all(args.seed)
    target = AutoModelForCausalLM.from_pretrained(args.target, dtype=torch.bfloat16,
        attn_implementation="sdpa").to(device).eval()
    draft = Qwen3DSparkModel.from_pretrained(args.draft, dtype=torch.bfloat16,
        attn_implementation="sdpa").to(device).eval()
    if target.config.model_type != "qwen3" or draft.markov_head is not None or draft.confidence_head is not None:
        parser.error("This diagnostic requires a Qwen3 DFlash checkpoint without Markov/confidence heads")
    assert_no_final_target_layer(target, draft.target_layer_ids)
    tokenizer = AutoTokenizer.from_pretrained(args.target)
    stop_ids = resolve_stop_token_ids(target, tokenizer)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    metadata = dict(args=vars(args) | {"output": str(args.output)}, device=torch.cuda.get_device_name(device),
        temperature=0, num_draft_layers=len(draft.layers), rollout_num_loops=draft.num_loops,
        timing_scope="median synchronized proposal+verification+context bookkeeping wall time; no prefill, probe reset, trace IO or gate overhead",
        interpretation="offline utility proxy on maximum-loop prefixes; oracle uses future information and is not deployed speedup")
    args.output.with_suffix(".metadata.json").write_text(json.dumps(metadata, indent=2))
    with args.output.open("w") as handle:
        for name in args.tasks.split(","):
            dataset = load_and_process_dataset(name, args.dataset_root)
            random.Random(args.seed).shuffle(dataset)
            for index, row in enumerate(dataset[:args.max_samples]):
                seed_all(args.seed + index)
                ids = encode_chat_messages(tokenizer, [{"role": "user", "content": row["turns"][0]}],
                    add_generation_prompt=True, enable_thinking=False).to(device)
                if index == 0:
                    for _ in range(args.warmup_samples):
                        scan_sample(target, draft, ids, stop_ids, args, name, index, None)
                    seed_all(args.seed + index)
                count = scan_sample(target, draft, ids, stop_ids, args, name, index, handle)
                print(f"{name} sample {index + 1}: {count} same-prefix rounds", flush=True)
    print(f"Trace: {args.output}")


if __name__ == "__main__":
    main()

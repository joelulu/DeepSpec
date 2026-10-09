"""Greedy same-prefix depth probes. Diagnostic timings are not serving TPS."""
from __future__ import annotations

import statistics
import time

import torch

from deepspec.eval.base_evaluator import has_stop_token, verify_draft_tokens
from deepspec.eval.dspark.draft_ops import build_dspark_proposal, forward_dspark_draft_block
from deepspec.modeling.dspark.common import extract_context_feature
from deepspec.modeling.dspark.context_cache import ContextKVCache


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.inference_mode()
def probe_depths(*, target, draft, target_cache, draft_cache, target_hidden_states,
                 output_ids, position_ids, start, stop_token_ids=None, repeats=3):
    """Compare all exits at one prefix and commit only the deepest exit.

    ContextKVCache.update/crop replace tensors, so a shallow dictionary snapshot
    isolates probes without copying the full context. Target cache rollback uses
    crop(start) before EVERY trial. The final deepest trial stays committed.
    """
    if repeats < 1 or not isinstance(draft_cache, ContextKVCache):
        raise ValueError("Need positive repeats and ContextKVCache")
    if target_cache.get_seq_length() != start:
        raise ValueError("Target KV must end immediately before the current anchor")
    device = output_ids.device
    block_size = int(draft.block_size)
    draft_input_ids = torch.full((1, block_size), int(draft.mask_token_id),
                                 dtype=torch.long, device=device)
    draft_input_ids[:, 0] = output_ids[:, start]
    rows = []
    selected = None
    previous_eval_loops = getattr(draft, "eval_num_loops", None)
    try:
        for loops in range(1, draft.num_loops + 1):
            timings = []
            previous_tokens = None
            for trial in range(repeats):
                target_cache.crop(start)
                cache = ContextKVCache()
                cache.layers = dict(draft_cache.layers)
                draft.eval_num_loops = loops
                sync(device)
                wall_start = time.perf_counter()
                events = [torch.cuda.Event(enable_timing=True) for _ in range(3)] if device.type == "cuda" else None
                if events:
                    events[0].record()
                hidden = forward_dspark_draft_block(draft,
                    draft_input_ids=draft_input_ids, position_ids=position_ids,
                    past_key_values_draft=cache, target_hidden_states=target_hidden_states,
                    start=start, block_size=block_size)
                proposal = build_dspark_proposal(draft, draft_input_ids=draft_input_ids,
                    block_hidden=hidden, block_size=block_size, temperature=0.0,
                    confidence_threshold=0.0)
                draft_end = time.perf_counter()
                if events:
                    events[1].record()
                verification = verify_draft_tokens(target_model=target, proposal=proposal,
                    position_ids=position_ids, start=start, past_key_values_target=target_cache,
                    temperature=0.0, max_proposal_tokens=block_size,
                    current_token_ids=output_ids[:, start:start + 1], stop_token_ids=stop_token_ids)
                verify_end = time.perf_counter()
                if events:
                    events[2].record()
                accepted = verification.accepted_draft_tokens
                if verification.terminated_by_stop_token:
                    committed = proposal.verify_input_ids[:, 1:accepted + 1]
                else:
                    committed = verification.committed_tokens
                progress = int(committed.shape[1])
                terminated = verification.terminated_by_stop_token or has_stop_token(committed, stop_token_ids)
                next_hidden = extract_context_feature(verification.target_output.hidden_states,
                    draft.target_layer_ids)[:, :accepted + 1]
                target_cache.crop(start + progress)
                cache.crop(start)
                sync(device)
                wall_ms = (time.perf_counter() - wall_start) * 1000
                draft_ms = events[0].elapsed_time(events[1]) if events else (draft_end - wall_start) * 1000
                verify_ms = events[1].elapsed_time(events[2]) if events else (verify_end - draft_end) * 1000
                timings.append((wall_ms, draft_ms, verify_ms))
                tokens = proposal.verify_input_ids[:, 1:].tolist()[0]
                signature = (tokens, accepted, committed.tolist())
                if previous_tokens is not None and signature != previous_tokens:
                    raise RuntimeError("Greedy probe changed across identical-prefix repeats")
                previous_tokens = signature
                # Last trial of the deepest exit is the only state retained.
                if loops == draft.num_loops and trial == repeats - 1:
                    selected = dict(committed_tokens=committed, target_hidden_states=next_hidden,
                        draft_cache=cache, terminated=terminated, accepted=accepted)
                else:
                    target_cache.crop(start)
            round_ms, draft_ms, verify_ms = (statistics.median(t[i] for t in timings) for i in range(3))
            rows.append(dict(num_loops=loops, accepted_draft_tokens=accepted,
                progress_tokens=progress, terminated=terminated, proposal_tokens=tokens,
                draft_ms=draft_ms, verify_ms=verify_ms, round_wall_ms=round_ms,
                progress_per_second_proxy=progress * 1000 / round_ms))
    finally:
        if previous_eval_loops is None:
            if hasattr(draft, "eval_num_loops"):
                del draft.eval_num_loops
        else:
            draft.eval_num_loops = previous_eval_loops
    return rows, selected

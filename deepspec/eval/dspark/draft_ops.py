from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers import DynamicCache

from deepspec.eval.base_evaluator import DraftProposal
from deepspec.eval.dspark.loopcd import LoopCDConfig, loopcd_logits
from deepspec.utils.sampling import logits_to_probs
from deepspec.modeling.dspark.gemma4 import Gemma4DSparkModel
from deepspec.modeling.dspark.qwen3 import Qwen3DSparkModel
from deepspec.modeling.dspark.context_cache import ContextKVCache


DSparkModel = Qwen3DSparkModel | Gemma4DSparkModel


@dataclass
class DSparkDraftProposal(DraftProposal):
    confidence_logits: torch.Tensor | None = None


def forward_dspark_draft_block(
    model: DSparkModel,
    *,
    draft_input_ids: torch.Tensor,
    position_ids: torch.Tensor,
    past_key_values_draft: DynamicCache | ContextKVCache,
    target_hidden_states: torch.Tensor,
    start: int,
    block_size: int,
    return_all_loop_hidden: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, ...]:
    draft_position_ids = position_ids[
        :, past_key_values_draft.get_seq_length() : start + block_size
    ]
    backbone_kwargs = {}
    if isinstance(model, Qwen3DSparkModel):
        backbone_kwargs["num_loops"] = getattr(model, "eval_num_loops", model.num_loops)
        if return_all_loop_hidden:
            backbone_kwargs["return_all_loop_hidden"] = True
    elif return_all_loop_hidden:
        raise ValueError("LoopCD exit collection currently supports Qwen3 only")
    block_hidden = model._forward_backbone(
        target_hidden_states=target_hidden_states,
        noise_embedding=model.embed_tokens(draft_input_ids),
        position_ids=draft_position_ids,
        attention_mask=None,
        past_key_values=past_key_values_draft,
        use_cache=True,
        is_causal=False,
        **backbone_kwargs,
    )
    if isinstance(past_key_values_draft, ContextKVCache):
        if any(pair[0].shape[-2] != start for pair in past_key_values_draft.layers.values()):
            raise RuntimeError("Context cache must end before the current anchor")
    past_key_values_draft.crop(start)
    return block_hidden


def _empty_dspark_proposal(draft_input_ids: torch.Tensor) -> DSparkDraftProposal:
    return DSparkDraftProposal(
        draft_token_count=0,
        verify_input_ids=draft_input_ids[:, :1],
        draft_probs=None,
        confidence_logits=None,
    )


def _predict_confidence_logits(
    model: DSparkModel,
    *,
    proposal_hidden_states: torch.Tensor,
    draft_input_ids: torch.Tensor,
    sampled_tokens: torch.Tensor,
    block_size: int,
) -> torch.Tensor | None:
    prev_token_ids = torch.cat(
        [draft_input_ids[:, :1], sampled_tokens[:, :-1]],
        dim=1,
    )
    confidence_pred = model.predict_confidence_step(
        proposal_hidden_states,
        prev_token_ids=prev_token_ids,
    )
    if confidence_pred is None:
        return None
    return confidence_pred.float().reshape(
        confidence_pred.shape[0],
        block_size,
        -1,
    )[:, :, 0]


def _confident_prefix_length(
    confidence_logits: torch.Tensor,
    *,
    block_size: int,
    threshold: float,
) -> int:
    if threshold <= 0.0:
        return int(block_size)
    below_threshold = confidence_logits.sigmoid() < threshold
    if not bool(below_threshold[0].any().item()):
        return int(block_size)
    return int(torch.nonzero(below_threshold[0], as_tuple=False)[0].item())


def build_dspark_proposal(
    model: DSparkModel,
    *,
    draft_input_ids: torch.Tensor,
    block_hidden: torch.Tensor,
    block_size: int,
    temperature: float,
    confidence_threshold: float,
    loopcd: LoopCDConfig | None = None,
    early_block_hidden: torch.Tensor | None = None,
) -> DSparkDraftProposal:
    assert draft_input_ids.size(0) == 1, "build_dspark_proposal requires batch_size=1"
    if (loopcd is None) != (early_block_hidden is None):
        raise ValueError("LoopCD config and early hidden states must be supplied together")
    proposal_hidden_states = block_hidden[:, :block_size, :]
    base_draft_logits = model.compute_logits(proposal_hidden_states)
    if loopcd is not None:
        if not isinstance(model, Qwen3DSparkModel):
            raise ValueError("LoopCD currently supports Qwen3 only")
        loopcd.validate_model(model, getattr(model, "eval_num_loops", model.num_loops))
        early_logits = model.compute_logits(early_block_hidden[:, :block_size, :])
        base_draft_logits = loopcd_logits(base_draft_logits, early_logits, loopcd)
    sampled_tokens, draft_logits = model.sample_draft_tokens(
        base_draft_logits,
        first_prev_token_ids=draft_input_ids[:, 0],
        temperature=temperature,
        hidden_states=proposal_hidden_states,
    )

    proposal_draft_tokens = int(block_size)
    confidence_logits = None
    if model.confidence_head is not None:
        confidence_logits = _predict_confidence_logits(
            model,
            proposal_hidden_states=proposal_hidden_states,
            draft_input_ids=draft_input_ids,
            sampled_tokens=sampled_tokens,
            block_size=block_size,
        )
        if confidence_logits is None:
            return _empty_dspark_proposal(draft_input_ids)
        proposal_draft_tokens = _confident_prefix_length(
            confidence_logits,
            block_size=block_size,
            threshold=float(confidence_threshold),
        )

    if proposal_draft_tokens == 0:
        return _empty_dspark_proposal(draft_input_ids)

    verify_input_ids = torch.cat(
        [draft_input_ids[:, :1], sampled_tokens[:, :proposal_draft_tokens]],
        dim=1,
    )
    draft_probs = logits_to_probs(
        draft_logits[:, :proposal_draft_tokens, :],
        temperature,
    )
    return DSparkDraftProposal(
        draft_token_count=proposal_draft_tokens,
        verify_input_ids=verify_input_ids,
        draft_probs=draft_probs,
        confidence_logits=(
            confidence_logits[:, :proposal_draft_tokens]
            if confidence_logits is not None
            else None
        ),
    )


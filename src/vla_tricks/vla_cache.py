"""Small compatibility helpers for the official VLA-Cache release."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from .perception import PatchReuseDecision


@torch.no_grad()
def visual_task_relevance(
    multihead_attention: Sequence[torch.Tensor],
    *,
    layer_id: int = 15,
    vision_token_start: int = 1,
    vision_token_count: int = 256,
) -> torch.Tensor:
    """Return one text-to-vision relevance value per visual patch."""
    if not multihead_attention:
        raise ValueError("at least one layer of attention is required")
    layer_id = min(layer_id, len(multihead_attention) - 1)
    attention = multihead_attention[layer_id].to(torch.float32)
    if attention.ndim != 4 or attention.shape[0] != 1:
        raise ValueError("expected attention shaped [1, heads, queries, keys]")

    mean_heads = attention[0].mean(dim=0)
    vision_end = vision_token_start + vision_token_count
    if mean_heads.shape[-1] < vision_end:
        raise ValueError("attention does not contain the expected visual tokens")
    text_start = min(vision_end, mean_heads.shape[-2] - 1)
    return mean_heads[text_start:, vision_token_start:vision_end].mean(dim=0)


def cache_positions_from_decision(
    decision: PatchReuseDecision, *, vision_token_start: int = 1
) -> list[int]:
    """Translate a patch decision into VLA-Cache sequence positions."""
    return [int(index) + int(vision_token_start) for index in decision.reusable_patch_ids]


@torch.no_grad()
def task_relevant_static_tokens(
    multihead_attention: Sequence[torch.Tensor],
    static_patch_ids: Sequence[int],
    *,
    layer_id: int = 15,
    top_k: int = 120,
    vision_token_start: int = 1,
    vision_token_count: int = 256,
) -> list[int]:
    """Return cache positions for static patches not important to the prompt.

    The released helper accidentally creates a boolean mask with the full
    attention-tensor shape and applies it to a two-dimensional matrix. Here the
    intended text-to-vision relevance is explicit: average heads, average query
    rows after the visual block, then rank visual keys by that score.
    """
    if top_k < 0:
        raise ValueError("top_k must be non-negative")
    relevance = visual_task_relevance(
        multihead_attention,
        layer_id=layer_id,
        vision_token_start=vision_token_start,
        vision_token_count=vision_token_count,
    )
    relevant_count = min(top_k, vision_token_count)
    relevant = set(
        torch.topk(relevance, k=relevant_count).indices.detach().cpu().tolist()
    )
    static = {
        int(index)
        for index in static_patch_ids
        if 0 <= int(index) < vision_token_count
    }
    return sorted(index + vision_token_start for index in static - relevant)

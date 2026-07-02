"""
Real KV-cached self-speculative decoding for SpatialVLA/Gemma2.

Same idea and guarantee as adaptive_sparse_vla/emu3_self_spec_decode.py (see
that file's docstring and experiments/SelfSpecDecode_univla.md for the full
rationale): draft with a few redundant decoder layers bypassed (the
DepthPruner mechanism from depth_prune_gemma2.py), verify with the full model,
accept the longest matching prefix -- output is byte-identical to plain greedy
decoding by construction, so this is a pure latency experiment with zero
accuracy risk. That property is exactly what SpatialVLA needs: static depth
pruning collapsed 3/4 tasks even at a single bypassed layer (see
docs/VISUAL_TOKENS_VS_LATENCY.md), so a lossless lever is the only depth-axis
option left worth trying here.

*** UNVERIFIED AGAINST THE REAL MODEL. As with the UniVLA version, this is
reasoned carefully from the transformers HybridCache API but has not run on
GPU. The first thing to check on Colab is losslessness. ***

Why this adapter is DIFFERENT from the UniVLA/Emu3 one (DynamicCache):
Gemma2 uses `HybridCache` -- a FIXED-SIZE, pre-allocated buffer written via
direct index assignment (`k_out[:, :, cache_position] = key_states`), not
DynamicCache's append/concat. Two consequences:
  1. `HybridCache` has no `.crop()` (confirmed: `hasattr(HybridCache, 'crop')
     == False` on transformers 4.51.3). We cannot "undo" a speculative write
     the way the Emu3 adapter does.
  2. Half of Gemma2's layers use SLIDING-WINDOW attention with a circular
     buffer (`_sliding_update`'s modulo-indexed shifting) that is only
     designed for monotonically-increasing, each-position-written-once
     access. Writing the SAME cache_position twice (e.g. draft then verify)
     is not verified safe for these layers.
  3. `HybridCache.get_seq_length()` is ONLY well-defined for `layer_idx=0`
     (raises for any other index) -- so layer 0 must NEVER be bypassed, or
     cache-position tracking silently breaks for every later forward call.
     `DepthPruner.apply_indices()` enforces this.

Design: **clone-then-promote**, never write any official-cache position more
than once.
  - DRAFT rolls out on a CLONE of the cache (cheap tensor .clone() per layer,
    no forward-pass cost beyond the draft's own cheap forwards); discarded
    after drafting.
  - VERIFY also runs on a CLONE of the (still pristine) official cache.
  - On full acceptance, the verify clone is exactly correct -> promote it to
    be the new official cache (no extra forward, same efficiency as the Emu3
    adapter's free-bonus-token path).
  - On a mismatch, the verify clone has wrong data past the accepted prefix,
    so it is discarded; ONE commit forward runs the corrected `chosen`
    sequence against the STILL-PRISTINE original official cache, writing
    each position exactly once.
Cost: cloning a HybridCache's per-layer tensors happens up to twice per round
(once for draft, once for verify) -- unlike Emu3's cheap `.crop()`, this is a
real memory-copy overhead that could eat into any latency win. Worth
measuring honestly, not assuming away.
"""
from __future__ import annotations

import copy
from typing import List, Optional, Sequence

import torch
from transformers.cache_utils import HybridCache


def _clone_hybrid_cache(cache: HybridCache) -> HybridCache:
    new = copy.copy(cache)
    new.key_cache = [k.clone() for k in cache.key_cache]
    new.value_cache = [v.clone() for v in cache.value_cache]
    return new


def _apply_processors(logits_processor, input_ids: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
    """See emu3_self_spec_decode.py::_apply_processors -- same
    context-independence assumption applies here."""
    if not logits_processor:
        return logits
    for proc in logits_processor:
        logits = proc(input_ids, logits)
    return logits


def _argmax1(logits_last_pos: torch.Tensor) -> int:
    return int(torch.argmax(logits_last_pos, dim=-1).item())


@torch.inference_mode()
def gemma2_self_speculative_generate(
    pruner,
    vla_model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    pixel_values,
    intrinsic,
    max_new_tokens: int,
    gamma: int,
    draft_layer_indices: Sequence[int],
    eos_token_id: Optional[int] = None,
    logits_processor: Optional[Sequence] = None,
    stats: Optional[dict] = None,
    cache_len_budget: int = 64,
) -> torch.Tensor:
    """Returns the full sequence (prompt + new tokens) as a (1, L) LongTensor.

    `pruner`: a `DepthPruner` (depth_prune_gemma2.py) wrapping
    `vla_model.language_model`, already calibrated (`.ranking` populated).
    `vla_model`: the SpatialVLA model (`policy.vla`/`policy.model`) --
    `forward()` handles vision-token merging internally and only needs
    `pixel_values`/`intrinsic` on the FIRST (prefill) call.
    `logits_processor`: same as `emu3_self_spec_decode.py` -- applied at
    every argmax decision so the lossless guarantee holds under the real
    generation constraints, not just in the abstract.

    `cache_len_budget`: the HybridCache is cloned up to twice per round (draft
    + verify) -- unlike DynamicCache's `.crop()`, this is a real memory-copy
    cost proportional to the cache's ALLOCATED size, not how much of it is
    actually used. `predict_action` calls `.generate(max_new_tokens=256, ...)`
    as a generous safety cap, but real action generations are ~12-30 tokens
    (see docs/VISUAL_TOKENS_VS_LATENCY.md's profiler measurements) -- sizing
    the cache for the ACTUAL realistic length instead of the full 256-token
    cap cuts every clone's cost by ~4-8x. If generation genuinely needs more
    than `cache_len_budget` new tokens, this raises (loud failure, not silent
    truncation) -- raise the budget if that happens on real data.
    """
    if gamma < 1:
        raise ValueError("gamma must be >= 1")
    device = input_ids.device
    dtype_ids = input_ids.dtype
    prompt_len = input_ids.shape[1]
    cache_new_tokens = min(max_new_tokens, max(cache_len_budget, gamma + 1))

    if stats is not None:
        stats.update(rounds=0, draft_calls=0, verify_calls=0, commit_calls=0,
                     accepted=0, proposed=0)

    def ones_mask(n):
        return torch.ones((1, n), device=device, dtype=attention_mask.dtype)

    gemma_config = pruner.lm.config
    cache = HybridCache(
        gemma_config, max_batch_size=1, max_cache_len=prompt_len + cache_new_tokens,
        device=device, dtype=vla_model.dtype,
    )

    pruner.restore()
    out = vla_model(input_ids=input_ids, attention_mask=attention_mask,
                    pixel_values=pixel_values, intrinsic=intrinsic,
                    past_key_values=cache, use_cache=True)
    cache = out.past_key_values or cache
    carry_logits = _apply_processors(logits_processor, input_ids, out.logits[:, -1, :].clone())
    if stats is not None:
        stats["verify_calls"] += 1

    full_ids = input_ids
    running_am = attention_mask
    confirmed_len = prompt_len
    n_new = 0

    def append_ids(id_list):
        nonlocal full_ids, running_am
        add = torch.tensor([id_list], device=device, dtype=dtype_ids)
        full_ids = torch.cat([full_ids, add], dim=1)
        running_am = torch.cat([running_am, ones_mask(len(id_list))], dim=1)

    while n_new < cache_new_tokens:
        if stats is not None:
            stats["rounds"] += 1

        free_token = _argmax1(carry_logits)
        round_tokens = [free_token]

        # --- DRAFT: on a CLONE, redundant layers bypassed ---
        n_draft = min(gamma - 1, cache_new_tokens - n_new - 1)
        eos_hit = eos_token_id is not None and free_token == eos_token_id
        if n_draft > 0 and not eos_hit:
            pruner.apply_indices(draft_layer_indices)
            d_cache = _clone_hybrid_cache(cache)
            d_am = torch.cat([running_am, ones_mask(1)], dim=1)
            cur_tok = torch.tensor([[free_token]], device=device, dtype=dtype_ids)
            for _ in range(n_draft):
                out = vla_model(input_ids=cur_tok, attention_mask=d_am,
                                pixel_values=None, intrinsic=None,
                                past_key_values=d_cache, use_cache=True)
                if stats is not None:
                    stats["draft_calls"] += 1
                d_cache = out.past_key_values or d_cache
                d_logits = _apply_processors(logits_processor, full_ids, out.logits[:, -1, :].clone())
                nxt = _argmax1(d_logits)
                round_tokens.append(nxt)
                cur_tok = torch.tensor([[nxt]], device=device, dtype=dtype_ids)
                d_am = torch.cat([d_am, ones_mask(1)], dim=1)
                if eos_token_id is not None and nxt == eos_token_id:
                    break
            pruner.restore()

        if stats is not None:
            stats["proposed"] += len(round_tokens) - 1

        # --- VERIFY: on a CLONE of the (still pristine) official cache ---
        v_cache = _clone_hybrid_cache(cache)
        verify_am = torch.cat([running_am, ones_mask(len(round_tokens))], dim=1)
        verify_input = torch.tensor([round_tokens], device=device, dtype=dtype_ids)
        out = vla_model(input_ids=verify_input, attention_mask=verify_am,
                        pixel_values=None, intrinsic=None,
                        past_key_values=v_cache, use_cache=True)
        if stats is not None:
            stats["verify_calls"] += 1
        target_logits = [
            _apply_processors(logits_processor, full_ids, out.logits[:, k, :].clone())
            for k in range(len(round_tokens))
        ]

        accepted = 1
        prev_logits = target_logits[0]
        for k in range(1, len(round_tokens)):
            if _argmax1(prev_logits) == round_tokens[k]:
                accepted += 1
                prev_logits = target_logits[k]
            else:
                break

        if stats is not None:
            stats["accepted"] += accepted - 1

        if accepted == len(round_tokens):
            # full acceptance: v_cache is exactly correct -> promote it,
            # no extra forward needed.
            chosen = round_tokens
            cache = out.past_key_values or v_cache
            carry_logits = target_logits[-1]
        else:
            # mismatch: v_cache has wrong data past `accepted` -- discard it.
            # Commit the corrected sequence against the STILL-PRISTINE
            # official cache in ONE forward (every position written once).
            correction = _argmax1(prev_logits)
            chosen = round_tokens[:accepted] + [correction]
            commit_am = torch.cat([running_am, ones_mask(len(chosen))], dim=1)
            commit_input = torch.tensor([chosen], device=device, dtype=dtype_ids)
            out = vla_model(input_ids=commit_input, attention_mask=commit_am,
                            pixel_values=None, intrinsic=None,
                            past_key_values=cache, use_cache=True)
            if stats is not None:
                stats["commit_calls"] += 1
            cache = out.past_key_values or cache
            carry_logits = _apply_processors(logits_processor, full_ids, out.logits[:, -1, :].clone())

        append_ids(chosen)
        confirmed_len += len(chosen)
        n_new += len(chosen)
        if eos_token_id is not None and eos_token_id in chosen:
            break
    else:
        if n_new >= cache_new_tokens and cache_new_tokens < max_new_tokens:
            print(f"[SpecDecode] WARNING: hit cache_len_budget ({cache_new_tokens}) without EOS -- "
                  f"sequence was cut shorter than plain generation would have produced. "
                  f"Raise --spec-decode-cache-budget.", flush=True)

    return full_ids

"""
Training-free self-speculative decoding for greedy (do_sample=False) action
generation.

Idea (per docs/VISUAL_TOKENS_VS_LATENCY.md): decode dominates latency (70-75%
of a control step) because it is many SEQUENTIAL single-token forwards through
the full LLM. Static depth pruning speeds each forward up but permanently
trades away accuracy (see docs/VISUAL_TOKENS_VS_LATENCY.md's SpatialVLA
section). Self-speculative decoding instead:

  1. DRAFT: run the SAME frozen model with a few redundant layers bypassed
     (our validated depth-pruning mechanism) to quickly propose `gamma` tokens.
  2. VERIFY: run the FULL (unpruned) model once, in parallel, over those
     `gamma` tokens; accept the longest prefix that matches what the full
     model would have greedily picked itself; replace the first wrong token
     (or append a free "bonus" token if all matched) with the full model's own
     choice.

Because verification always has final say, the final token sequence is
BYTE-IDENTICAL to what the full model would have produced generating token-by-
token on its own (see `test_self_spec_decode.py`::test_losslessness). Whether
this reduces *wall-clock* depends on the acceptance rate vs. draft/verify
overhead -- that must be measured on the real model, this module only
guarantees correctness.

No training, no external module: draft and verify share every weight: only
the active layer subset differs, toggled via caller-supplied `mode`.

This module is deliberately KV-cache-agnostic: `logits_fn(token_ids, mode)`
takes the full token history and a mode string ("draft"|"verify") and returns
next-token logits, however the caller wants to compute them (recompute from
scratch, or internally use incremental KV caching for real speed). This keeps
the accept/reject/bonus logic in this file testable on CPU with no GPU/model
dependency; `emu3_self_spec_decode.py` implements the real, KV-cached
`logits_fn` for UniVLA/Emu3.
"""
from __future__ import annotations

from typing import Callable, List, Optional, Sequence

LogitsFn = Callable[[Sequence[int], str], Sequence[float]]


def _argmax(logits: Sequence[float]) -> int:
    best_i, best_v = 0, logits[0]
    for i, v in enumerate(logits):
        if v > best_v:
            best_i, best_v = i, v
    return best_i


def greedy_self_speculative_decode(
    logits_fn: LogitsFn,
    prompt_ids: Sequence[int],
    max_new_tokens: int,
    gamma: int,
    eos_token_id: Optional[int] = None,
    stats: Optional[dict] = None,
) -> List[int]:
    """Generate up to `max_new_tokens` new tokens after `prompt_ids`, greedily,
    using self-speculative decoding. Returns the NEW tokens only (not the
    prompt). Guaranteed identical to calling `greedy_decode` below.

    If `stats` is passed (an empty dict), it is filled with
    {"rounds": int, "draft_calls": int, "verify_calls": int, "accepted": int,
    "proposed": int} so callers can compute acceptance rate / estimate whether
    the draft/verify overhead is actually worth it on the real model."""
    if gamma < 1:
        raise ValueError("gamma must be >= 1")
    if stats is not None:
        stats.update(rounds=0, draft_calls=0, verify_calls=0, accepted=0, proposed=0)

    tokens = list(prompt_ids)
    new_tokens: List[int] = []
    # carry_logits = target/full model's next-token distribution given `tokens`
    # as they currently stand (no un-verified draft tokens appended yet).
    carry_logits = logits_fn(tokens, "verify")
    if stats is not None:
        stats["verify_calls"] += 1

    while len(new_tokens) < max_new_tokens:
        if stats is not None:
            stats["rounds"] += 1
        # --- DRAFT: propose up to gamma tokens with the shallow model ---
        draft_tokens: List[int] = []
        draft_ctx = list(tokens)
        for _ in range(min(gamma, max_new_tokens - len(new_tokens))):
            dl = logits_fn(draft_ctx, "draft")
            if stats is not None:
                stats["draft_calls"] += 1
            d = _argmax(dl)
            draft_tokens.append(d)
            draft_ctx.append(d)
            if eos_token_id is not None and d == eos_token_id:
                break
        if not draft_tokens:
            break
        if stats is not None:
            stats["proposed"] += len(draft_tokens)

        # --- VERIFY: one pass of the full model over all draft tokens ---
        # NOTE: this reference implementation calls logits_fn once per draft
        # token because the logits_fn(tokens)->single-position abstraction
        # can't express "one batched forward returning gamma positions at
        # once". The real GPU adapter (emu3_self_spec_decode.py) does this
        # verification as ONE forward pass over all gamma positions -- that
        # parallelism is exactly what makes speculative decoding fast; this
        # toy harness only exists to test accept/reject/bonus correctness.
        verify_ctx = list(tokens)
        target_logits_seq: List[Sequence[float]] = []
        for d in draft_tokens:
            verify_ctx.append(d)
            target_logits_seq.append(logits_fn(verify_ctx, "verify"))
            if stats is not None:
                stats["verify_calls"] += 1

        # --- accept the longest matching prefix ---
        accepted = 0
        prev_logits = carry_logits
        for k, d in enumerate(draft_tokens):
            if _argmax(prev_logits) == d:
                accepted += 1
                prev_logits = target_logits_seq[k]
            else:
                break

        if accepted == len(draft_tokens):
            # full acceptance: take a free bonus token from the target model
            bonus = _argmax(target_logits_seq[-1])
            chosen = draft_tokens + [bonus]
        else:
            # first mismatch at index `accepted`: use the TARGET's own choice
            # instead of the (wrong) drafted token.
            correction = _argmax(prev_logits)  # target dist right before the mismatch
            chosen = draft_tokens[:accepted] + [correction]

        if stats is not None:
            stats["accepted"] += accepted

        # carry_logits for the NEXT round must reflect the ACCEPTED sequence
        # (which may end in a corrected token the verify pass never actually
        # saw as input) -- always recompute fresh, do not reuse
        # target_logits_seq entries computed from a rejected draft token.
        carry_logits = logits_fn(list(tokens) + chosen, "verify")
        if stats is not None:
            stats["verify_calls"] += 1

        for t in chosen:
            if len(new_tokens) >= max_new_tokens:
                break
            tokens.append(t)
            new_tokens.append(t)
            if eos_token_id is not None and t == eos_token_id:
                return new_tokens

    return new_tokens


def greedy_decode(
    logits_fn: LogitsFn,
    prompt_ids: Sequence[int],
    max_new_tokens: int,
    eos_token_id: Optional[int] = None,
) -> List[int]:
    """Plain one-token-at-a-time greedy decoding with the FULL model
    (mode="verify" always). Reference implementation for the losslessness
    test: self-speculative decoding must always produce this exact sequence."""
    tokens = list(prompt_ids)
    new_tokens: List[int] = []
    while len(new_tokens) < max_new_tokens:
        logits = logits_fn(tokens, "verify")
        t = _argmax(logits)
        tokens.append(t)
        new_tokens.append(t)
        if eos_token_id is not None and t == eos_token_id:
            break
    return new_tokens

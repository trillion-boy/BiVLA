"""
Real KV-cached self-speculative decoding for UniVLA/Emu3.

See self_spec_decode.py for the backbone-agnostic algorithm and its CPU-proven
losslessness (test_self_spec_decode.py). This module wires that algorithm to
the real model, reusing the existing depth-pruning infra
(`policy._apply_llm_pruning` / `policy._restore_llm_layers`) as the training-
free "draft" (bypass a few redundant layers) vs "verify" (full model) toggle.

*** UNVERIFIED AGAINST THE REAL MODEL -- this is reasoned from the Emu3/HF
cache API (DynamicCache.crop, standard causal-LM forward signature) that the
rest of this project's pruning code already relies on, but has not been run
on GPU. The very first thing to check on Colab is losslessness: run
`smoke_test_losslessness()` (or the standalone script) and confirm
spec-decode output is token-for-token IDENTICAL to plain generation before
trusting any latency number. Expect this to need at least one debugging
round, the same as FastV (RoPE re-indexing) and depth pruning (DynamicCache
gaps) did. ***

Design ("free first token" per round): each round's first proposed token
comes for free from the PREVIOUS round's full-model logits (`carry_logits`)
-- by construction this is exactly what the full model would pick, so it
never needs verification. Only tokens 2..gamma of a round are produced by the
cheap draft (pruned) forward and get checked against the full model. This
avoids an extra "redo last token in draft mode" forward per round.

Per round:
  1. free_token = argmax(carry_logits)                      [0 forward calls]
  2. draft gamma-1 more tokens, redundant layers bypassed    [gamma-1 draft forwards]
  3. crop cache back to confirmed_len (discard the draft pass's KV writes --
     both real ones from surviving layers and the BypassDecoderLayer
     placeholder zeros from bypassed layers)
  4. verify: ONE full-model forward over all `gamma` round tokens           [1 verify forward]
  5. accept the longest prefix matching what the full model's own argmax
     says (free_token always matches trivially); on a mismatch, crop away
     the rejected tail and take the full model's own correction token; on
     full acceptance, the last verify logits directly become next round's
     carry_logits (no extra forward needed).
"""
from __future__ import annotations

from typing import List, Optional, Sequence

import torch
from transformers.cache_utils import DynamicCache


def _apply_processors(logits_processor, input_ids: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
    """Apply a list of HF-style LogitsProcessor callables (input_ids, scores)
    -> scores, e.g. `_generate_sequences`'s `ActionIDConstraintLogitsProcessor`.
    That processor is context-INDEPENDENT (masks a fixed allowed-token set
    regardless of input_ids), so `input_ids` here need only be *a* valid
    sequence tensor, not necessarily the exact partial context -- if a
    context-dependent processor is ever introduced this assumption must be
    revisited (each verify position would need its own exact prefix)."""
    if not logits_processor:
        return logits
    for proc in logits_processor:
        logits = proc(input_ids, logits)
    return logits


def _argmax1(logits_last_pos: torch.Tensor) -> int:
    """logits_last_pos: (1, vocab) -> python int."""
    return int(torch.argmax(logits_last_pos, dim=-1).item())


@torch.inference_mode()
def emu3_self_speculative_generate(
    policy,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    max_new_tokens: int,
    gamma: int,
    draft_layer_indices: Sequence[int],
    eos_token_id: Optional[int] = None,
    logits_processor: Optional[Sequence] = None,
    stats: Optional[dict] = None,
) -> torch.Tensor:
    """Returns the full sequence (prompt + new tokens) as a (1, L) LongTensor,
    matching the shape/content of `outputs.sequences` from
    `policy.model.generate(...)` so `_decode_action_sequences` works
    unchanged on the result.

    `draft_layer_indices`: layer indices to bypass during drafting -- reuse
    a previously-calibrated redundancy ranking (e.g.
    `policy._depth_ranking[:4]`), same mechanism validated in
    docs/DEPTH_PRUNING_RESULTS.md. `policy` must be a UniVLAInference-like
    object exposing `.model`, `._apply_llm_pruning(indices)`,
    `._restore_llm_layers()`.

    `logits_processor`: the SAME list passed to `model.generate(...)` in
    `_generate_sequences` (e.g. `[self._action_id_processor]`) -- MUST be
    applied here too, at every argmax decision (free token, draft tokens,
    verify comparisons, correction, bonus), or the "lossless vs. plain
    greedy decoding" guarantee breaks (draft/verify could pick tokens outside
    the constrained action vocabulary that plain generation never would).
    """
    if gamma < 1:
        raise ValueError("gamma must be >= 1")
    model = policy.model
    device = input_ids.device
    dtype_ids = input_ids.dtype

    if stats is not None:
        stats.update(rounds=0, draft_calls=0, verify_calls=0, accepted=0, proposed=0)

    def ones_mask(n):
        return torch.ones((1, n), device=device, dtype=attention_mask.dtype)

    # --- prefill once with the FULL model ---
    policy._restore_llm_layers()
    cache = DynamicCache()
    out = model(input_ids=input_ids, attention_mask=attention_mask,
                past_key_values=cache, use_cache=True)
    cache = out.past_key_values
    carry_logits = _apply_processors(logits_processor, input_ids, out.logits[:, -1, :].clone())
    if stats is not None:
        stats["verify_calls"] += 1

    full_ids = input_ids  # grows each round; the returned sequence
    running_am = attention_mask  # grows in lockstep with full_ids
    confirmed_len = full_ids.shape[1]
    n_new = 0

    def append_ids(id_list):
        nonlocal full_ids, running_am
        add = torch.tensor([id_list], device=device, dtype=dtype_ids)
        full_ids = torch.cat([full_ids, add], dim=1)
        running_am = torch.cat([running_am, ones_mask(len(id_list))], dim=1)

    while n_new < max_new_tokens:
        if stats is not None:
            stats["rounds"] += 1

        free_token = _argmax1(carry_logits)
        round_tokens = [free_token]

        # --- DRAFT: gamma-1 more tokens, redundant layers bypassed ---
        n_draft = min(gamma - 1, max_new_tokens - n_new - 1)
        eos_hit_in_draft = (eos_token_id is not None and free_token == eos_token_id)
        if n_draft > 0 and not eos_hit_in_draft:
            policy._apply_llm_pruning(draft_layer_indices)
            d_cache = cache
            d_am = torch.cat([running_am, ones_mask(1)], dim=1)
            cur_tok = torch.tensor([[free_token]], device=device, dtype=dtype_ids)
            for _ in range(n_draft):
                out = model(input_ids=cur_tok, attention_mask=d_am,
                            past_key_values=d_cache, use_cache=True)
                if stats is not None:
                    stats["draft_calls"] += 1
                d_cache = out.past_key_values
                d_logits = _apply_processors(logits_processor, full_ids, out.logits[:, -1, :].clone())
                nxt = _argmax1(d_logits)
                round_tokens.append(nxt)
                cur_tok = torch.tensor([[nxt]], device=device, dtype=dtype_ids)
                d_am = torch.cat([d_am, ones_mask(1)], dim=1)
                if eos_token_id is not None and nxt == eos_token_id:
                    break
            policy._restore_llm_layers()

        if stats is not None:
            stats["proposed"] += len(round_tokens) - 1  # free_token isn't speculative

        # --- VERIFY: discard the draft pass's KV writes, redo with the full model ---
        cache.crop(confirmed_len)
        verify_am = torch.cat([running_am, ones_mask(len(round_tokens))], dim=1)
        verify_input = torch.tensor([round_tokens], device=device, dtype=dtype_ids)
        out = model(input_ids=verify_input, attention_mask=verify_am,
                    past_key_values=cache, use_cache=True)
        if stats is not None:
            stats["verify_calls"] += 1
        # target_logits[k] = full model's (processor-constrained) prediction
        # for the token AFTER round_tokens[k] (i.e. it "checks" round_tokens[k+1]).
        target_logits = [
            _apply_processors(logits_processor, full_ids, out.logits[:, k, :].clone())
            for k in range(len(round_tokens))
        ]

        accepted = 1  # free_token is always accepted by construction
        prev_logits = target_logits[0]
        for k in range(1, len(round_tokens)):
            if _argmax1(prev_logits) == round_tokens[k]:
                accepted += 1
                prev_logits = target_logits[k]
            else:
                break

        if stats is not None:
            stats["accepted"] += accepted - 1  # free_token doesn't count

        if accepted == len(round_tokens):
            # full acceptance: cache already correctly covers all of
            # round_tokens (verify wrote real KV for every position) -- no
            # extra forward needed, next round's carry_logits is free.
            chosen = round_tokens
            carry_logits = target_logits[-1]
        else:
            # mismatch at index `accepted`: drop the rejected tail's KV,
            # take the full model's own choice, and do one small forward to
            # both write its KV and obtain the next round's carry_logits.
            cache.crop(confirmed_len + accepted)
            correction = _argmax1(prev_logits)
            corr_am = torch.cat([running_am, ones_mask(accepted + 1)], dim=1)
            out = model(
                input_ids=torch.tensor([[correction]], device=device, dtype=dtype_ids),
                attention_mask=corr_am,
                past_key_values=cache, use_cache=True,
            )
            if stats is not None:
                stats["verify_calls"] += 1
            cache = out.past_key_values
            carry_logits = _apply_processors(logits_processor, full_ids, out.logits[:, -1, :].clone())
            chosen = round_tokens[:accepted] + [correction]

        append_ids(chosen)
        confirmed_len += len(chosen)
        n_new += len(chosen)
        if eos_token_id is not None and eos_token_id in chosen:
            break

    return full_ids

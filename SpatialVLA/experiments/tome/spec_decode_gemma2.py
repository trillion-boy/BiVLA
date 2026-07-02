"""
Patch `vla_model.generate` to route through self-speculative decoding
(gemma2_self_spec_decode.py) instead of HF's built-in `.generate()`.

`SpatialVLAForConditionalGeneration.predict_action()` calls
`self.generate(**model_inputs, max_new_tokens=256, do_sample=False)` and
expects a plain (1, L) token-id tensor back (no `return_dict_in_generate`) --
our adapter already returns exactly that shape/type, so this patch is a
drop-in swap for the common case (single-sequence greedy decoding, which is
the ONLY mode this project ever uses: do_sample=False everywhere).

We cannot read `simpler_env`'s installed source in this environment (it is
only present on Colab, not checked into this repo), so the exact kwarg names
`predict_action` passes to `.generate()` are inferred from
`SpatialVLAForConditionalGeneration.forward()`'s signature
(input_ids/attention_mask/pixel_values/intrinsic) rather than verified
directly. `verbose=True` prints the kwarg keys on the first call specifically
so this can be checked immediately on Colab; any call that doesn't match the
expected simple pattern (missing input_ids/attention_mask, sampling, beams,
multiple return sequences) safely falls back to the ORIGINAL `.generate()` --
never silently wrong, at worst silently not-speculative.
"""
from __future__ import annotations

from typing import Optional, Sequence

from gemma2_self_spec_decode import gemma2_self_speculative_generate


def apply_gemma2_self_spec_decode(
    vla_model,
    pruner,
    gamma: int,
    draft_layer_indices: Sequence[int],
    eos_token_id: Optional[int] = None,
    logits_processor: Optional[Sequence] = None,
    cache_len_budget: int = 64,
    verbose: bool = True,
):
    if getattr(vla_model, "_specdec_patched", False):
        remove_gemma2_self_spec_decode(vla_model)
    vla_model._specdec_orig_generate = vla_model.generate
    vla_model._specdec_stats = {}
    vla_model._specdec_seen_first_call = False

    def patched_generate(*args, **kwargs):
        if verbose and not vla_model._specdec_seen_first_call:
            print(f"[SpecDecode] first .generate() call kwargs: {sorted(kwargs.keys())} "
                  f"(args: {len(args)})", flush=True)
            vla_model._specdec_seen_first_call = True

        input_ids = kwargs.get("input_ids", args[0] if args else None)
        attention_mask = kwargs.get("attention_mask")
        pixel_values = kwargs.get("pixel_values")
        intrinsic = kwargs.get("intrinsic")
        do_sample = kwargs.get("do_sample", False)
        gen_cfg = kwargs.get("generation_config")
        if gen_cfg is not None:
            do_sample = getattr(gen_cfg, "do_sample", do_sample)
        num_beams = kwargs.get("num_beams", 1)
        num_return_sequences = kwargs.get("num_return_sequences", 1)
        max_new_tokens = kwargs.get("max_new_tokens", 256)

        unsupported = (
            input_ids is None or attention_mask is None
            or do_sample or num_beams > 1 or num_return_sequences > 1
        )
        if unsupported:
            if verbose:
                print("[SpecDecode] call pattern not supported (sampling/beams/missing "
                      "input) -- falling back to plain .generate()", flush=True)
            return vla_model._specdec_orig_generate(*args, **kwargs)

        return gemma2_self_speculative_generate(
            pruner, vla_model, input_ids, attention_mask, pixel_values, intrinsic,
            max_new_tokens=max_new_tokens, gamma=gamma, draft_layer_indices=draft_layer_indices,
            eos_token_id=eos_token_id, logits_processor=logits_processor,
            stats=vla_model._specdec_stats, cache_len_budget=cache_len_budget,
        )

    vla_model.generate = patched_generate
    vla_model._specdec_patched = True
    if verbose:
        print(f"[SpecDecode] patched .generate(): gamma={gamma}, draft layers={list(draft_layer_indices)} "
              f"(lossless -- output identical to plain greedy decoding by construction).", flush=True)
    return vla_model


def remove_gemma2_self_spec_decode(vla_model):
    if getattr(vla_model, "_specdec_patched", False):
        vla_model.generate = vla_model._specdec_orig_generate
        del vla_model._specdec_orig_generate
        vla_model._specdec_patched = False
    return vla_model

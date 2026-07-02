# Self-speculative decoding (UniVLA) — lossless latency, not an accuracy trade

**Motivation.** Static depth pruning (`LLM_PRUNE_COUNT`, the depth controller) always
trades some accuracy for latency — on SpatialVLA/Gemma2 that trade turned out
very steep (see `docs/VISUAL_TOKENS_VS_LATENCY.md`'s Gemma2 section: even a
single bypassed layer hurt 3/4 tasks). Self-speculative decoding is a different
kind of lever: it uses the SAME redundant-layer-bypass mechanism, but only as a
fast **draft** whose proposals are always checked by the full model before being
accepted. **The final output is byte-identical to plain greedy decoding with the
full model, by construction** — proven on CPU (see below), not just claimed.
This makes it a pure latency experiment with zero accuracy risk, unlike every
other lever tried so far.

## How it works (training-free, no external module)
Per control step's action-token generation, per "round" of up to `gamma` tokens:
1. **Free token.** The first token of the round is exactly `argmax` of the full
   model's own logits carried over from the previous round (or the prefill) — no
   compute needed, and it can never be "wrong" since it *is* the full model's
   choice.
2. **Draft.** The other `gamma-1` tokens are proposed by the SAME frozen model
   with a few redundant decoder layers bypassed (the same cosine-redundancy
   ranking validated for depth pruning) — cheap, sequential, single-token forwards.
3. **Verify.** One parallel forward of the FULL model over all `gamma` round
   tokens. Accept the longest prefix that matches what the full model's own
   argmax would have picked; on a mismatch, take the full model's own correction
   token instead of the (wrong) draft; on full acceptance, get one bonus token
   for free.
4. Cache surgery: `DynamicCache.crop()` discards the draft pass's speculative KV
   writes before verification recomputes that span with the full model — so the
   cache is always exactly what plain generation would have produced.

## Correctness (proven, not assumed)
- `self_spec_decode.py` + `test_self_spec_decode.py`: the accept/reject/bonus
  algorithm, cache-agnostic, proven lossless vs. plain greedy across
  disagree-rates 0.0–0.9 and gamma 1–8.
- `emu3_self_spec_decode.py` + `test_emu3_self_spec_decode.py`: the REAL
  KV-cached adapter, tested against a tiny but real causal transformer using
  the actual `transformers.cache_utils.DynamicCache` and the actual
  `BypassDecoderLayer` class (moved to `bypass_layer.py` so both the real
  inference stack and this CPU test share one implementation). Proven lossless
  across 15 seeds × 4 gamma values × several draft-layer subsets × EOS handling
  × a constrained-vocabulary logits processor (mirroring
  `ActionIDConstraintLogitsProcessor`, which spec-decode must also respect to
  stay lossless).

**What is NOT yet verified:** the real Emu3 model's exact forward interface
(kwargs, RoPE quirks) on GPU. Every other real-model integration this session
(FastV, depth-pruning DynamicCache gaps) needed at least one Colab debugging
round after passing CPU tests — expect the same here. **The first thing to
check on Colab, before any latency claim, is losslessness**: run baseline vs.
spec-decode with `SPEC_DECODE_ENABLE=1` and diff the action outputs — they must
be identical.

## Implementation
- `adaptive_sparse_vla/self_spec_decode.py` — backbone-agnostic core algorithm.
- `adaptive_sparse_vla/emu3_self_spec_decode.py` — real KV-cached adapter.
- `adaptive_sparse_vla/bypass_layer.py` — `BypassDecoderLayer`, shared by static
  pruning, the depth controller, and this.
- `inference.py`: `_maybe_calibrate_spec_decode` (reuses the depth controller's
  ranking if already calibrated this episode), the `_generate_sequences` branch
  that calls the adapter instead of `.generate()`, `pruning_summary()` reports
  `spec_decode.acceptance_rate`.

Gated by env: `SPEC_DECODE_ENABLE=1`, `SPEC_DECODE_GAMMA` (default 4),
`SPEC_DECODE_LAYER_COUNT` (default 4, how many top-redundant layers the draft
bypasses).

## How to run (Colab, `bivla` env)

**Step 1 — losslessness check (do this first, small N):**
```bash
!N_EPISODES=2 bash /content/run_univla.sh widowx_put_eggplant_in_basket baseline /content/spec_base
!N_EPISODES=2 SPEC_DECODE_ENABLE=1 SPEC_DECODE_GAMMA=4 SPEC_DECODE_LAYER_COUNT=4 \
  bash /content/run_univla.sh widowx_put_eggplant_in_basket baseline /content/spec_on
```
Compare the per-episode action plans / success — they must match `spec_base`
exactly (same env_ids, same seed). Any divergence means a real-model interface
bug in `emu3_self_spec_decode.py`, not an accuracy trade-off (there shouldn't be
one) — fix before measuring latency.

**Step 2 — latency measurement (once lossless is confirmed), N=24:**
```bash
!N_EPISODES=24 bash /content/run_univla.sh widowx_put_eggplant_in_basket baseline /content/spec_base_n24
!N_EPISODES=24 SPEC_DECODE_ENABLE=1 SPEC_DECODE_GAMMA=4 SPEC_DECODE_LAYER_COUNT=4 \
  bash /content/run_univla.sh widowx_put_eggplant_in_basket baseline /content/spec_on_n24
```
Compare `avg_model_ms_per_infer` and check `spec_decode.acceptance_rate` in the
per-episode JSON (high acceptance = most rounds get all `gamma` tokens for
"draft-speed" cost — that's what would make this actually faster; low
acceptance means verify overhead ate the gain, worth trying a smaller
`SPEC_DECODE_LAYER_COUNT` or `SPEC_DECODE_GAMMA`).

### What to look at (the hypothesis)
- **Success rate**: must be identical to baseline (this is guaranteed by
  construction, not a hypothesis — if it differs, something is broken).
- **Latency**: unlike static pruning, this is genuinely uncertain and worth
  measuring honestly — action sequences are short (12–30 tokens), a less
  favorable regime for speculative decoding than the long sequences it usually
  shines on. If acceptance rate is high but latency doesn't improve, the
  verify-pass overhead is likely eating the gain at this sequence length.

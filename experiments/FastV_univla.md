# FastV — training-free visual-token pruning on UniVLA's frozen Emu3 LLM

**Goal (project core).** Cut inference latency on a *frozen* UniVLA without
training and without any external module (no GroundingDINO), keeping task
success — by pruning redundant visual tokens *inside* the Emu3 LLM. The
importance signal is the model's **own attention**, so there is nothing extra to
run. "Foveation feel": task-relevant tokens keep full attention through every
layer; the periphery is dropped after an early layer.

## Why FastV here (and not ToMe)
UniVLA/Emu3 has **no ViT**. Images become a fixed rectangular grid of **discrete
VQ tokens** whose size is declared in the text prefix, then the Emu3 LLM reads
them. You cannot average two discrete IDs, and dropping any breaks the declared
grid → reducing tokens at the *input* is structurally out-of-distribution (this
is exactly why the repo's "compact focus" used blur, which kept the token count
and so never reduced latency). The latency actually lives in the **LLM** chewing
through every visual token at every layer and every decode step. FastV cuts that.

## How it works (`adaptive_sparse_vla/fastv_emu3.py`)
1. **Layers `0..K-1` run on the full grid** so the model aggregates the scene
   (early layers carry the spatial structure).
2. **At layer `K`**, score each visual token by the attention the generation
   cursor (last position) pays it — the model's own attention, no extra model.
   Keep all non-visual tokens + the top `keep_ratio` of visual tokens; **drop the
   rest from the hidden stream** for all layers `> K`. The discrete input grid is
   never touched → no input OOD; pruning happens in latent space after the scene
   is understood → OOD-tolerant on the frozen model.
3. The dropped tokens **never enter the deep layers' KV cache**, so the many
   autoregressive action-decode steps attend to a much shorter cache → real
   wall-clock savings.

### Correctness details (the parts that are easy to get wrong)
- **Survivors keep their ORIGINAL positions** (no re-indexing). Re-indexing kept
  tokens to `0..M-1` destroys the model's learned spatial grounding — the action
  tokens attend to specific grid offsets via RoPE, so renumbering makes them look
  at the wrong place and the policy stops grasping (observed: 0% success). To keep
  original positions we patch Emu3's rotary embedding to return its **full
  precomputed cos/sin table** instead of slicing to the (shorter) key length, so
  `cos[position_ids]` is valid for the original, non-contiguous positions. This is
  transparent for unpruned layers (same position-absolute values).
- After a pruned prefill the KV cache is **heterogeneous**: early layers hold `N`
  entries, deep layers hold `M`. The patched forward drives the decode itself,
  giving every layer `attention_mask=None`. Because positions are kept original,
  the new token's absolute position is the same for all layers (= layer 0's cache
  length), so decode needs just one position id. (Delegating to the stock forward
  would build one shared mask of the wrong width and crash.)
- Pruning fires **only on the generation prefill** (`q_len > 1` with
  `use_cache`); decode and the no-cache layer-pruning calibration pass through.

## Unit test (CPU, no model/GPU)
`adaptive_sparse_vla/test_fastv_logic.py` validates the bookkeeping that the
math depends on:
- `visual_mask_from_input_ids` marks exactly the `img_token … eof/eoi` span;
- keep-set keeps every non-visual token + top-`keep_ratio` visual + the cursor;
- a pruned prefill shortens the sequence and preserves all non-visual tokens;
- a decode step passes through;
- **prefill→decode with a heterogeneous cache**: early layers cache `N`, deep
  layers cache `M`, and several decode steps grow each layer from its own length
  with no shape errors.

```bash
cd adaptive_sparse_vla && python test_fastv_logic.py
# -> ALL FASTV LOGIC TESTS PASSED
```

## How to run the eval (Colab, `bivla` env, transformers 4.51.3)
FastV is wired into `adaptive_sparse_vla/inference.py` and gated purely by env
vars — no CLI change. Run the **same** UniVLA task baseline vs FastV and compare
`avg_elapsed` (latency) and `success_rate`.

```python
import os
# baseline (FastV off): just run as usual.

# FastV on:
os.environ["FASTV_ENABLE"]     = "1"
os.environ["FASTV_K"]          = "3"     # prune after 3 full layers (2–4 typical)
os.environ["FASTV_KEEP_RATIO"] = "0.4"   # keep 40% of visual tokens past layer K
run("widowx_carrot_on_plate", "shared", "/content/results/carrot_fastv", n=24)

# turn off afterwards
for k in ("FASTV_ENABLE", "FASTV_K", "FASTV_KEEP_RATIO"):
    os.environ.pop(k, None)
```

### What to look at
- `avg_elapsed` — should **drop** with FastV (the latency win), more so with
  smaller `FASTV_KEEP_RATIO` / smaller `FASTV_K`.
- `success_rate` — should **hold** vs baseline. If it drops: raise `keep_ratio`,
  raise `K` (let the model aggregate longer before pruning).
- Sweep: `K ∈ {2, 3, 4}`, `keep_ratio ∈ {0.25, 0.4, 0.6}`.

### Notes / next step
- Unlike the SpatialVLA ToMe variant (which restores the token count), FastV
  keeps the sequence short through the deep layers, so it cuts **both** the deep
  layers and every decode step — the bigger lever for UniVLA, where the LLM is
  the whole cost.
- Stacks with **temporal KV caching** (planned next): reuse unchanged visual
  tokens' KV across frames so even the prefill shrinks over time. FastV (spatial)
  and temporal caching are orthogonal.
- The importance signal can later be swapped for AutoGaze saliency; the default
  uses the LLM's own attention (zero external module — the project constraint).

# ToMe — training-free token merging on SpatialVLA's frozen SigLIP ViT

**Goal.** Reduce inference latency on a *frozen* SpatialVLA (PaliGemma2 = SigLIP +
Gemma2) **without training** and **without pushing the backbone out of
distribution**, while holding task success on SimplerEnv / WidowX-Bridge.

## Why SpatialVLA and not UniVLA
UniVLA (Emu3) encodes images with a **discrete VQ tokenizer**: the visual tokens
are codebook IDs laid out as a fixed rectangular `H×W` grid whose size is written
into the text prefix (`processing_emu3.py`). You cannot average two discrete IDs,
and dropping any breaks the declared grid — so token reduction there is
**structurally OOD** (this is exactly why the repo's "compact focus" uses *blur*,
which keeps the token count and therefore never reduced latency).

SpatialVLA uses a real **SigLIP ViT** with continuous patch tokens
(`modeling_spatialvla.py: get_image_features` → `vision_tower(...).last_hidden_state`).
ToMe (ICLR'23) was designed for off-the-shelf frozen ViTs, so it fits here
natively.

## How it stays OOD-safe (the design)
1. **Merge, don't drop.** Between SigLIP encoder layers we fuse the `r` most
   similar tokens by weighted average (`bipartite_soft_matching`). Redundant
   background patches collapse; distinctive patches have no similar neighbour and
   survive — "merge the background, keep the important region sharp".
2. **Unmerge at the end.** After the last merged layer we broadcast each merged
   cluster back to all of its original patch positions, so the tensor handed to
   the projector / Gemma2 has the **same token count and grid layout** as the
   baseline. The language model sees nothing out of distribution.
3. **Latency win** comes from the encoder's middle layers running on fewer
   tokens (real ViT FLOPs cut). The downstream LLM cost is unchanged in this
   variant (count restored) — a deliberate, safe first step.
4. **Optional protection.** Pass a per-patch importance map; protected patches
   are never merged away nor absorb a merge. Built-in `center` prior keeps a
   centred square at full resolution (WidowX objects are roughly centred) with
   zero added latency and no extra model.

## Files
- `tome_siglip.py` — self-contained (torch-only) ToMe merge/unmerge + a patcher
  `apply_tome_to_siglip(vision_tower, r, num_merge_layers, protect_provider)` and
  `remove_tome(...)`. Also `center_protect_provider(keep_ratio)`.
- `test_tome_logic.py` — CPU unit test (no model/GPU). Verifies: token count
  preserved end-to-end, protected tokens bit-exact, redundant tokens collapse,
  a patched (fake) SigLIP tower returns the original token count.

Run the unit test:
```bash
cd SpatialVLA/experiments/tome && python test_tome_logic.py
# -> ALL TOME LOGIC TESTS PASSED
```

## How to run the eval (Colab, `spatialvla` env)
The flags are wired into `SpatialVLA/experiments/latent_saccade/spatialvla_eval.py`.
Compare baseline vs ToMe on the **same** task; watch `ms/step` (latency) and
`성공률` (success). Disable latent mask / DINO so you measure ToMe alone.

```bash
EVAL=SpatialVLA/experiments/latent_saccade/spatialvla_eval.py
COMMON="--model-path <SPATIALVLA_CKPT> --task widowx_put_eggplant_in_basket \
        --n-episodes 24 --no-latent-mask"

# (a) baseline — ToMe off
conda run -n spatialvla python $EVAL $COMMON \
    --output-dir /content/results/eggplant_baseline

# (b) ToMe, pure similarity
conda run -n spatialvla python $EVAL $COMMON \
    --tome --tome-r 8 --tome-layers 6 --tome-protect none \
    --output-dir /content/results/eggplant_tome_pure

# (c) ToMe, centre-protected (keep middle 25% sharp)
conda run -n spatialvla python $EVAL $COMMON \
    --tome --tome-r 8 --tome-layers 6 --tome-protect center --tome-protect-ratio 0.25 \
    --output-dir /content/results/eggplant_tome_center
```

### What to look at
- `ms/step` — should **drop** with ToMe (the latency win). Larger `--tome-r` /
  more `--tome-layers` → faster but more approximate.
- `성공률` / `파지율` — should **hold** vs baseline (the constraint). If success
  drops, lower `r`, fewer merge layers, or switch `--tome-protect center`.
- Sweep suggestion: `r ∈ {4, 8, 12}`, `layers ∈ {4, 6, 9}`, protect ∈ {none, center}.

### Notes / next steps
- This variant restores the token count → only the **ViT** is cheaper. SigLIP is
  the smaller half of SpatialVLA, so expect a **modest** latency cut. To also cut
  the Gemma2 prefill we would *not* unmerge — but SpatialVLA's tokens encode
  spatial position, so reducing the final count risks spatial-reasoning OOD;
  that's a separate, riskier experiment.
- Next planned lever (orthogonal, stackable): **temporal ViT-feature caching** —
  reuse static patch features across frames, recompute only changed ones.
- Importance map upgrade: replace the `center` prior with an AutoGaze saliency
  map so protection is content-driven rather than a fixed centre box.

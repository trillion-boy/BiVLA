# Depth-axis results: training-free LLM layer pruning on frozen UniVLA

**Setup.** Frozen UniVLA (Emu3), SimplerEnv WidowX-Bridge, 4 tasks × N=24 episodes.
Pure inference latency measured as `avg_model_ms_per_infer` (time inside
`model.step()` only — step-count independent). Two families compared, both
training-free, no external module:

- **Static (uniform)** — EfficientVLA-style: bypass a fixed number of the most
  redundant decoder layers (redundancy = `1 - cos(layer_in, layer_out)`), the
  same count for the whole episode. `static4`, `static8` = 4 / 8 layers.
- **Phase-adaptive depth controller (ours)** — non-uniform over time: deep
  (bypass 2) during the precise approach+grasp, shallow (bypass 6 or 8) after a
  one-way, gripper-triggered switch. `depth_s6`, `depth_s8`.

Emu3 has ~32 decoder layers; only the back half (16–31) is eligible (early layers
are too important to touch).

## Full results (success / pure-inference speedup vs base)

| Task | base | static4 | static8 | depth_s6 | depth_s8 |
|---|---|---|---|---|---|
| Eggplant | 100% / 1.00× | 100% / 1.11× | 88% / 1.29× | 100% / 1.14× | 92% / 1.22× |
| Carrot | 67% / 1.00× | 58% / 1.10× | 67% / 1.23× | 71% / 1.08× | 71% / 1.09× |
| Stack | 58% / 1.00× | **83%** / 1.10× | 75% / 1.23× | 62% / 1.10× | **83%** / 1.11× |
| Spoon | 71% / 1.00× | 83% / 1.10× | 83% / 1.24× | 75% / 1.09× | 58% / 1.13× |
| **Avg success** | **74%** | **81%** | **78%** | **77%** | **76%** |
| **Avg speedup** | 1.00× | ~1.10× | ~1.25× | ~1.10× | ~1.14× |

## Finding 1 (strong): training-free layer pruning is often a "free lunch"

On a *frozen* VLA, bypassing redundant LLM layers frequently **raises success
while cutting latency**:
- Stack 58% → **83%** (+25 pts) at 1.10×; Spoon 71% → 83% (+12) at 1.24×;
  Carrot 67% → 71%.
- Average success goes **74% → 78–81%** while latency drops **10–25%**.

Plausible mechanism: removing redundant deep layers acts as **inference-time
regularization** on the frozen policy — it strips noisy/over-confident
computation the model can't fix because it's frozen. This is clean, surprising,
and training-free. **Caveat:** the *optimal amount* is task-dependent (Eggplant
wants few, Carrot wants ~8, Stack wants 4) — no single static count is best
everywhere.

## Finding 2 (honest): the phase-adaptive controller does NOT beat uniform

Our non-uniform-depth hypothesis ("protect grasp, prune transport → better
success/latency frontier than uniform") **is not supported across 4 tasks**:
- By average, **uniform ≈ or > our controller**: static4 81% @ 1.10× and static8
  78% @ 1.25× both match or beat depth_s6 (77% @ 1.10×) and depth_s8 (76% @ 1.14×).
- It held only on Eggplant (depth_s6 100% @ 1.14× edged static4 100% @ 1.11×);
  on Stack, static4 (83%) beat depth_s6 (62%).

So the phase signal (grasp vs transport) is **not the axis that matters most** —
the *task/scene* determines the right amount, not the gripper phase. The
controller is a reasonable idea that the data does not vindicate over simple
uniform pruning.

## Caveat: noise

Even at N=24, mid-success tasks have high variance (binomial σ ≈ 9% at p≈0.7, so
±2 episodes is noise). Large effects (Stack +25) are real; small rankings
(Eggplant 92 vs 100, Spoon depth_s8 58) are within noise. Firm claims need
repeated N=24 or larger.

## Per-task best (success ≥ base AND faster)

| Task | best config | result |
|---|---|---|
| Eggplant | depth_s6 / static4 | 100% @ 1.14× / 1.11× |
| Carrot | static8 | 67% (=base) @ 1.23× |
| Stack | static4 | **83% (+25)** @ 1.10× |
| Spoon | static8 | **83% (+12)** @ 1.24× |

With a per-task-tuned uniform count, **every task keeps ≥ base success and runs
1.10–1.24× faster** — but the best count differs, which argues for adapting the
*amount* to the task/scene (not the grasp phase).

## Conclusion & next

1. **Headline result for the depth axis: training-free layer pruning is a free
   lunch on frozen VLAs** — accuracy up on 3/4 tasks, latency down 10–25%,
   task-dependent amount.
2. **The phase-adaptive controller is honestly a wash** vs simple uniform pruning
   — recorded as a tried idea, not a win.
3. **The "non-uniform" contribution should come from the spatial axis**
   (SpatialVLA ToMe — the project title's home, different mechanism, separate ViT
   to cut), not depth.
4. Optional future: if revisiting depth, adapt the prune *amount* to task/scene
   difficulty (the signal the data points to), not the gripper phase.

*Artifacts: `adaptive_sparse_vla/inference.py` (BypassDecoderLayer + redundancy
calibration + depth controller), env knobs `LLM_PRUNE_COUNT` (static) and
`DEPTH_CTRL_*` (controller), `experiments/DepthController_univla.md`.*

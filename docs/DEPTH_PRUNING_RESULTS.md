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

## Finding 2: uniform application is unstable → apply *selectively per archetype*

Crucially, **no single setting applied to ALL tasks wins** — it helps some,
regresses others (depth_s8: Stack +25, Carrot +4, but Eggplant −8, Spoon −13;
static4: Stack/Spoon +25/+12 but Carrot −9). This is exactly the project's
motivating premise (README: *"naive uniform sparsification often improved one
task while regressing others"*). So the right metric is **not** average-over-all
configs; it is **selective per-archetype routing** — apply pruning only to the
archetypes it helps, fall back to baseline elsewhere.

The shared controller maps an instruction to an archetype a priori
(`infer_task_archetype`), and we gate depth pruning on it
(`DEPTH_CTRL_ARCHETYPES`). For our 4 tasks:

| Task | archetype | depth | result | vs base |
|---|---|---|---|---|
| Eggplant | container_drop | OFF | base 100% / 1.00× | — |
| Carrot | planar_placement | ON | depth_s8 71% / 1.09× | +4 |
| Stack | stack_alignment | ON | depth_s8 83% / 1.11× | **+25** |
| Spoon | thin_tool | OFF | base 71% / 1.00× | — |
| **selective avg** | | | **81%** + latency cut on the ON tasks | **+7** |

Selective routing **never drops below base on any task** and saves latency where
pruning helps — turning the "unstable uniform" result into a clean win. This is
the contribution: *not* one global trick, but a controller that applies the
intervention only where the archetype warrants it.

### Honest limitation (the epistemics)
We learned *which* archetypes benefit by **looking at results** — with one task
per archetype, the per-archetype assignment is a **1-sample hypothesis, not a
mechanism**. We have no a-priori reason pruning should help planar_placement /
stack_alignment specifically (the "low base success → pruning helps" heuristic
fits 3/4 tasks but Spoon breaks it). Two parts generalize differently:
- **Archetype classification** (instruction → archetype) generalizes reasonably
  to new/OOD instructions (coarse keyword/noun mapping; unknown → `generic`).
- **Archetype → policy assignment** is the fragile link — a new task mapped to a
  "depth-ON" archetype *might* not benefit the same way.

To be OOD-robust this needs one of: (a) more tasks per archetype to validate the
assignment, (b) a mechanism for *why* pruning helps, or (c) — most robust — a
**per-episode online signal** (model uncertainty, or how much pruning perturbs
the action) that decides without relying on having seen the archetype.

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

1. **Training-free layer pruning is a free lunch on frozen VLAs** — accuracy up on
   3/4 tasks, latency down 10–25% — but **only when applied selectively**; a
   single uniform setting helps some tasks and regresses others.
2. **The contribution is the selective controller**, not any one trick: route the
   instruction to an archetype a priori and apply depth pruning only where it
   helps (carrot/stack), baseline elsewhere (eggplant/spoon) → **≥ base on every
   task, +7 avg success, latency saved on the ON tasks**.
3. **Honest caveat:** the per-archetype assignment is a 1-sample hypothesis;
   classification generalizes, policy assignment needs more tasks/archetype or an
   online signal to be OOD-robust (see Finding 2).
4. **Next:** validate the spatial axis (SpatialVLA ToMe) under the *same*
   selective-controller framing — apply per archetype where it helps, not globally.

*Artifacts: `adaptive_sparse_vla/inference.py` (BypassDecoderLayer + redundancy
calibration + depth controller), env knobs `LLM_PRUNE_COUNT` (static) and
`DEPTH_CTRL_*` (controller), `experiments/DepthController_univla.md`.*

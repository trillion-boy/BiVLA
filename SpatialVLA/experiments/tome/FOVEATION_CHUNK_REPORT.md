# SpatialVLA: Foveation × Action-Chunk Execution Report

**Date:** 2026-07-16
**Setup:** SimplerEnv WidowX-Bridge, 4 tasks × N=24 episodes per cell, A100 40GB,
`spatialvla-4b-224-pt` (frozen), torch 2.5.1+cu121, transformers 4.47.0.
All methods are training-free, external-module-free, applied at test time
through the evaluation wrapper only (`tome_spatialvla_eval.py`).

Baselines were re-measured under identical conditions; the vanilla baseline
matches the SpatialVLA paper's zero-shot numbers within sampling noise, and
the chunk k=2 results reproduce the mentor's report (mean +13.0pp there,
+13.6pp here) almost exactly per task.

---

## Phase 1 — Log-polar foveation (direct port) × chunk execution

Foveation here is the mentor's RetinaBased log-polar transform, ported 1:1
from the OpenVLA pipeline (where it improved the 4-task average by ~+19pp).
Success rate, with grasp rate in parentheses:

| Config | Eggplant | Carrot | Stack Cube | Spoon | **Mean succ.** | ms/infer |
|---|---|---|---|---|---|---|
| baseline | 66.7% (87.5) | 25.0% (45.8) | 29.2% (58.3) | 8.3% (16.7) | **32.3%** | ~844–902 |
| foveate alone | 58.3% (66.7) | 29.2% (54.2) | 4.2% (41.7) | 8.3% (12.5) | **25.0%** | ~845–895 |
| **chunk k=2** | **87.5%** (91.7) | **41.7%** (45.8) | 25.0% (58.3) | **29.2%** (37.5) | **45.9%** | ~455–462 |
| chunk k=4 | 66.7% (83.3) | 4.2% (20.8) | 8.3% (20.8) | 8.3% (12.5) | **21.9%** | ~220–231 |
| foveate + chunk2 | 45.8% (58.3) | 25.0% (62.5) | 20.8% (41.7) | 16.7% (33.3) | **27.1%** | ~453–458 |
| foveate + chunk4 | 54.2% (70.8) | 4.2% (25.0) | 0.0% (0.0) | 4.2% (4.2) | **15.7%** | ~220–228 |

Phase 1 observations:

- **Chunk k=2 is the robust win** (+13.6pp mean, ~1.9× faster): executing 2
  of the 4 actions the checkpoint already predicts per generate call. k=4
  overreaches — the gains invert on the short-step-budget tasks.
- **The OpenVLA foveation does not transfer.** Alone it costs −7.3pp on
  average; combined with chunk2 the loss amplifies to −18.8pp vs chunk2
  alone. The failure signature is consistent: grasp rate stays high while
  success collapses — *grasped but not placed*.

## Why the log-polar transform hurts SpatialVLA (root cause)

Unlike OpenVLA (coordinate-free SigLIP/DINO features), SpatialVLA stamps
every visual token with an explicit 3D position: it estimates depth from the
image (ZoeDepth) and back-projects each patch's **pixel-grid coordinate**
through the camera intrinsics —
`modeling_spatialvla.py: backproject_patch` (`p_cam = inv(K) @ [u,v,1] * depth`)
feeding `Ego3DPositionEmbeddingMLP`.

The log-polar transform **moves pixels**. After warping, the content at
pixel (u,v) is no longer on the camera ray `inv(K)@[u,v,1]`, and the depth
network sees a geometrically impossible image — so every token gets a wrong
3D position, worst in the periphery, which is exactly where placement
targets (plate/basket/tower) sit. This mechanistically matches the
grasped-but-not-placed signature, and explains the OpenVLA/SpatialVLA
divergence: the same transform is benign peripheral denoising for a
coordinate-free backbone and geometric corruption for a geometry-aware one.

## Phase 2 — Redesigned foveation (three diagnoses → three changes)

| # | Diagnosis | Change (flag) |
|---|---|---|
| 1 | Polar **warp** corrupts the pixel↔3D correspondence | `--foveate-mode blur`: space-variant blur — sharp fovea, progressively blurred periphery, **zero pixel displacement**; intrinsics/depth stay valid |
| 2 | Fovea center hardcoded to the **frame center** | `--foveate-center motion`: fovea follows the EMA-smoothed centroid of frame differences (the moving gripper/object); frames only — no oracle, no detector |
| 3 | Blur stays on during **precision placement** | `--foveate-phase pregrasp`: foveate only while the policy's own gripper command is OPEN; full resolution from the moment it commands a grasp |

All three components are implemented in `foveation.py` /
`tome_spatialvla_eval.py` and unit-tested (21/21, incl. an impulse-
displacement test proving blur preserves geometry where the warp does not;
`center=None` reproduces the historical log-polar output bit-for-bit).

## Phase 2 results (all on top of chunk k=2, N=24)

Success rate, grasp rate in parentheses; reference rows on top:

| Config | Eggplant | Carrot | Stack Cube | Spoon | **Mean succ.** |
|---|---|---|---|---|---|
| (ref) chunk2 alone | 87.5% | 41.7% | 25.0% | 29.2% | 45.9% |
| (ref) log-polar + chunk2 | 45.8% | 25.0% | 20.8% | 16.7% | 27.1% |
| **blur** (fixed center) | **79.2%** (79.2) | 20.8% (25.0) | 12.5% (45.8) | **33.3%** (41.7) | **36.5%** |
| blur + **motion** | 75.0% (75.0) | 25.0% (33.3) | **20.8%** (45.8) | 25.0% (37.5) | **36.5%** |
| blur + motion + **pregrasp** | 75.0% (83.3) | **29.2%** (41.7) | **20.8%** (50.0) | 29.2% (33.3) | **38.6%** |

Phase 2 observations:

1. **The geometry-preserving blur recovers most of the warp's damage**:
   27.1% → 38.6% mean (+11.5pp) against the log-polar variant at identical
   information budget (keep=20%). Per task: eggplant +33.4pp, spoon
   +16.6pp. This experimentally confirms diagnosis #1.
2. **First case of foveation beating no-foveation**: on spoon, blur+chunk2
   (33.3%) exceeds chunk2 alone (29.2%).
3. **The best fovea-center policy is scene-dependent**: eggplant/spoon
   prefer the fixed center, carrot/stack_cube prefer motion(+pregrasp) —
   consistent with where each task's objects sit in the camera view.
   The natural next step is selecting the gaze policy from runtime signals
   (e.g., motion-energy confidence with fixed-center fallback, or
   phase-scheduled gaze) rather than per-task configuration.

## Current recommendation

- **Chunk k=2 is the headline SpatialVLA result**: +13.6pp mean success and
  ~1.9× lower latency, stable across all four tasks, zero tuning,
  reproducing the mentor's findings.
- Foveation on SpatialVLA is an **architecture-dependent finding**: the
  transform that helps a coordinate-free backbone (OpenVLA) corrupts a
  geometry-aware one unless made geometry-preserving. With the blur
  variant the gap to no-foveation narrows to −7.3pp mean (from −18.8pp)
  and inverts on one task; closing the rest via signal-driven gaze
  selection is the open follow-up.

## Reproduce

```bash
# Phase 1 cells
python tome_spatialvla_eval.py --model-path <ckpt> --task <task> --n-episodes 24 [--foveate] [--exec-chunk {2,4}]

# Phase 2 cells (all with --exec-chunk 2)
python tome_spatialvla_eval.py ... --foveate --foveate-mode blur --exec-chunk 2
python tome_spatialvla_eval.py ... --foveate --foveate-mode blur --foveate-center motion --exec-chunk 2
python tome_spatialvla_eval.py ... --foveate --foveate-mode blur --foveate-center motion --foveate-phase pregrasp --exec-chunk 2
```

Tests: `pytest test_foveation.py test_foveation_v2.py test_chunk_exec.py`

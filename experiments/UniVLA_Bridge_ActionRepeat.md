# UniVLA on SimplerEnv WidowX-Bridge — action repeat 2

Run of 2026-08-05, `UNIVLA_SIMPLER_BRIDGE_VIDEO_BS128_20K`, `--model-type
baseline --action-repeat 2`, 4 tasks × N=24 = 96 episodes, `--image-size 256
--min-pixels 6400`, Colab L4.

## The flag did what it was supposed to

UniVLA predicts a 5-action chunk per forward (`predict_action_frames = 5`,
`adaptive_sparse_vla/inference.py:768`) and the baseline executes all five, so
`--action-repeat 2` should put it at **10 env steps per model call**. The
measured ratio confirms it:

| task | ms/infer | ms/env-step | ratio |
|---|---|---|---|
| carrot | 2895 | 291 | **9.95** |
| eggplant | 2664 | 269 | **9.90** |

The same ratio was ~5 on the baseline campaign (2826 ms/infer, 603 ms/env-step).

## Results

| task | baseline ¹ | action repeat 2 | Δ | grasp base → repeat |
|---|---|---|---|---|
| eggplant | 100.0 | **16.7** | −83.3 | 100.0 → **83.3** |
| stack | 75.0 | 4.2 | −70.8 | 100.0 → 12.5 |
| spoon | 70.8 | 0.0 | −70.8 | 75.0 → 4.2 |
| carrot | 66.7 | 8.3 | −58.4 | 66.7 → 16.7 |
| **avg** | **78.1%** | **7.3%** (7/96) | **−70.8** | 85.4 → **29.2** |

¹ The 2026-07-20 chunk-exec/foveation campaign's baseline. Three UniVLA/Bridge
baselines exist (82.3% / 78.1% / 74.0%); see the note in
`LabMeeting_4Backbone_Summary.md`. **The choice does not matter here** — the
effect is −65 to −75 against any of them, and eggplant is 100% in all three.

An unpaired z on the average is ≈14. A paired test would only make it larger.

## The failure is "never finishes", not "finishes wrong"

Episode lengths sit at the step cap almost everywhere:

| task | baseline avg steps | repeat 2 avg steps | cap |
|---|---|---|---|
| carrot | 34.4 | **58** | 60 |
| stack | 34.6 | **59** | 60 |
| spoon | 31.1 | **60** | 60 |
| eggplant | 26.2 | **112** | 120 |

The policy is not making a wrong choice and committing to it; it never reaches a
terminal state at all.

## Why: horizon and displacement move together

SimplerEnv's control mode here is `arm_pd_ee_target_delta_pose_align2` — a
**relative** action space. Repeating an action therefore does two things at once,
and they are easy to conflate:

1. **Open-loop horizon doubles** — 5 → 10 env steps between observations.
2. **Commanded displacement doubles** — the target pose advances twice as far per
   decision.

So this is not a clean "hold the same command longer" intervention; it is "go
twice as far with half the feedback". Overshoot is the expected outcome and it is
what the step counts show.

Eggplant is the informative exception: **grasp holds at 83.3% while success falls
to 16.7%**. It still reaches and closes on the object, then fails to place it. The
other three tasks fail earlier, at the grasp. Eggplant's step cap is 120 rather
than 60, which gives the approach phase twice the budget to recover from
overshoot — consistent with the displacement account rather than a
task-difficulty one.

## What this contributes

The same flag, the same code, applied at the same hook point:

| backbone / benchmark | baseline horizon | under repeat 2 | Δ |
|---|---|---|---|
| SpatialVLA / Bridge | 1 | 2 | **+10.4** |
| UniVLA / Bridge | 5 | **10** | **−70.8** |
| UniVLA / LIBERO | 10 | 20 | **−68.0** |

**The honest reading is horizon, not architecture.** UniVLA did not collapse
because it is Emu3-based; it collapsed because its baseline already executed a
5-action chunk, so the same flag landed it at 10 steps. Any chunking policy would
be in the same position. The two UniVLA rows (5→10 and 10→20) give nearly the
same −70, which suggests the damage is already saturated by 10 steps rather than
scaling with the horizon.

This is exactly why the grid uses action repeat rather than chunk-exec: it is the
one temporal operation that is *identical* across backbones. The consequence is
that the resulting horizons are *not* matched, so **steps/call must be reported
next to every action-repeat row** or the comparison reads as backbone-dependence
when it is horizon-dependence.

## What would separate horizon from displacement

Neither is isolated by the runs so far. Two cheap follow-ups:

* **SpatialVLA action repeat 4 and 5.** Puts a non-chunking backbone at 4 and 5
  env steps per call, spanning the gap between SpatialVLA's +10.4 at 2 and
  UniVLA's −70.8 at 10, on a single backbone. If it degrades smoothly the story
  is horizon; if it holds to 5 and then falls, the threshold sits between 5 and
  10 and the backbones agree.
* **UniVLA `--exec-chunk 5 --action-repeat 2` vs `--exec-chunk 10`** — not
  runnable as-is (the chunk is only 5), but the equivalent on a longer-chunk
  policy would hold the horizon fixed while varying displacement.

## Reproduce

```bash
python adaptive_sparse_vla/eval.py \
  --emu-hub  <UNIVLA_SIMPLER_BRIDGE_VIDEO_BS128_20K> \
  --vq-hub   <Emu3-VisionTokenizer> \
  --fast-path <fast_bridge_t5_s50> \
  --task widowx_carrot_on_plate --n-episodes 24 \
  --image-size 256 --min-pixels 6400 --device cuda \
  --model-type baseline --action-repeat 2
```

Per-episode JSONs: `results/univla_bridge_0805/action_repeat2/`.

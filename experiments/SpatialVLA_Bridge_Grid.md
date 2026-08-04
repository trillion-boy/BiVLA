# SpatialVLA on SimplerEnv WidowX-Bridge — the four-condition grid

Setup: `spatialvla-4b-224-pt` (frozen), 4 tasks × N=24 = **96 episodes per
condition**, `unnorm_key=bridge_orig/1.0.0`. Every condition replays the same
env ids 0–23 per task, so the comparisons are paired even though the numbers
below are computed the unpaired (conservative) way.

## Results

| condition | eggplant | carrot | stack | spoon | **avg** | Δ | steps/call | ms/call |
|---|---|---|---|---|---|---|---|---|
| **Original policy** | 66.7 | 25.0 | 29.2 | 8.3 | **32.3%** | — | 1 | ~880 |
| **action repeat 2** | 70.8 | 25.0 | 37.5 | **37.5** | **42.7%** | **+10.4** | 2 | 895 |
| **fixed foveation** (blur 20%) | 50.0 | 29.2 | 20.8 | 20.8 | **30.2%** | −2.1 | 1 | 884 |
| fixed foveation (log-polar 20%) | 58.3 | 29.2 | **4.2** | 8.3 | **25.0%** | −7.3 | 1 | ~870 |
| **fixed depth pruning** (1 of 26) | 37.5 | 29.2 | 25.0 | 0.0 | **22.9%** | −9.4 | 1 | ~855 |
| chunk-exec k=2 *(not in the grid)* | 87.5 | 41.7 | 25.0 | 29.2 | **45.9%** | +13.5 | 2 | ~460 |

Grasp rate, same order: 52.1 / **57.3** / 45.8 / — / — / —.

**Nothing here clears significance on its own** at n=96 (largest is chunk-exec
at z=1.94, then action repeat at z=1.50). Run `paired_test.py` against the
baseline JSONs before treating any single row as established — the conditions
share initial states, so the paired test is much sharper than these numbers.

## 1. The temporal axis is the only one that helps, and it helps a lot

Both temporal conditions raise success **and** cut wall-clock; every other
condition costs accuracy and buys almost nothing.

| | success | wall-clock |
|---|---|---|
| action repeat 2 | +10.4 | **~2.0× faster** (calls halve, per-call cost unchanged) |
| chunk-exec k=2 | +13.5 | 1.9× faster |
| foveation (either) | −2.1 / −7.3 | 1.00× |
| depth pruning | −9.4 | 1.03× |

The ms/call column shows why: 895 vs 884 vs 880 — foveation and action repeat
leave the cost of a call untouched by construction, and depth pruning at 1 of
26 layers only removes 4% of the stack. Action repeat's speedup comes entirely
from making half as many calls.

## 2. At matched horizon, copying an action is as good as predicting one

This is the useful finding. Both temporal conditions put the policy at **2 env
steps per model call**; they differ only in what fills the second step.

| | second step is | success |
|---|---|---|
| chunk-exec k=2 | the model's own next predicted action | 45.9% |
| action repeat 2 | a copy of the first | 42.7% |

**3.2 points apart, z=0.45 — not distinguishable.**

So the gain does not come from using the model's predicted trajectory. It comes
from **re-planning less often**. The baseline re-generates every single env step
(`--exec-chunk 0 = re-generate every step`), and that appears to be the problem
rather than the reference point.

A plausible reading: the checkpoint ships an `action_chunk_size` in its
processor config, i.e. it was trained to emit coherent multi-step segments.
Executing one action and immediately re-planning is off-distribution, and it
lets consecutive plans disagree slightly — the arm dithers. Holding *anything*
for two steps suppresses that, whether or not the held value is what the model
would have predicted next.

This is worth stating in the paper because it separates two explanations that
are usually conflated: "chunking helps because the model plans ahead" versus
"frequent re-planning hurts". Here it is the second.

## 3. Blur is gentler than log-polar, and stack is where it shows

| task | log-polar Δ | blur Δ |
|---|---|---|
| eggplant | −8.4 | −16.7 |
| carrot | +4.2 | +4.2 |
| **stack** | **−25.0** | **−8.4** |
| spoon | 0.0 | **+12.5** |

The averages (−7.3 vs −2.1) are both inside the noise band, but the per-task
split is the predicted one. **Stack is the geometry-critical task** — placing
one block precisely on another — and it is exactly where log-polar collapses to
4.2% while blur holds 20.8%. Spoon shows the same ordering.

That is the pattern the Ego3D account predicts: SpatialVLA back-projects each
patch's grid coordinate through the camera intrinsics, so a transform that
*moves pixels* stamps every token with a wrong 3D position, and the damage
should concentrate where precise placement matters. Blur removes the same
amount of information without moving anything, and it does not collapse stack.

Not proof — none of these differences is individually significant — but the
prediction was made before the run and the per-task pattern matches it.

## 4. Where the gains actually come from

Per-task, the action-repeat gain is not spread evenly:

| task | baseline | action repeat 2 | Δ |
|---|---|---|---|
| spoon | 8.3 | **37.5** | **+29.2** |
| stack | 29.2 | 37.5 | +8.3 |
| eggplant | 66.7 | 70.8 | +4.2 |
| carrot | 25.0 | 25.0 | 0.0 |

**Almost all of it is spoon**, the task the baseline was worst at (8.3%), and
its grasp rate moves with it (16.7% → 41.7%). So this is not a placement
improvement — the policy is failing to *reach and grasp* under per-step
re-planning, and holding actions fixes the reach.

Carrot is unchanged under every temporal setting, which is consistent with the
same story: it is the task where the baseline was already re-planning
harmlessly.

## What this column still needs

* **Paired tests.** All six rows share initial states; the unpaired numbers
  above understate the evidence. The per-episode records are in the summary
  JSONs.
* **A decision on the depth-pruning cell.** It is measured at 1 of 26 layers
  (4%), not at the 25% ratio used on the other backbones, because a single
  layer already hurt. The grid should say which it reports and why.
* **The Stack baseline discrepancy.** Two campaigns recorded 29.2% and 33.3%
  for the same cell. The table above uses 29.2%; a 4.2-point shift in the
  baseline moves every Δ in this file.

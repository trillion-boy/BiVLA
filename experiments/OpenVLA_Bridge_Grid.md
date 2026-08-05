# OpenVLA on SimplerEnv WidowX-Bridge — baseline, action repeat, depth pruning

Run of 2026-08-05, `openvla/openvla-7b`, `unnorm_key=bridge_orig`, 4 tasks ×
N=24 = 96 episodes per condition, Colab. All three conditions ran back to back
in one session on one GPU, so the deltas are within-campaign and the episodes
are matched.

## Results

| condition | eggplant | carrot | stack | spoon | **avg** | Δ | grasp | ms/infer | ms/env-step |
|---|---|---|---|---|---|---|---|---|---|
| **Original policy** | 25.0 | 16.7 | 12.5 | 8.3 | **15.6%** | — | 28.1 | 518 | 518 |
| action repeat 2 | 12.5 | 4.2 | 8.3 | 4.2 | **7.3%** | −8.3 | 22.9 | 519 | **260** |
| fixed depth pruning (1 of 32) | **41.7** | 12.5 | 12.5 | 4.2 | **17.7%** | +2.1 | 34.4 | 504 | 504 |

The baseline reproduces the 15.6% already recorded for OpenVLA/Bridge in §5.1
of the reports.

## Paired tests

`adaptive_sparse_vla/paired_test.py`, 96 matched episodes each.

**Neither condition is distinguishable from chance.**

| | discordant pairs | McNemar exact p |
|---|---|---|
| action repeat 2 | 14 (11 broke a success, 3 fixed a failure) | **0.057** |
| depth pruning 1 | 16 (7 broke, **9 fixed**) | **0.80** |

The depth row deserves emphasis because the aggregate is misleading. +2.1 points
looks like a small win and the eggplant column (25.0 → 41.7) looks like a large
one, but episode by episode the condition fixed 9 baseline failures and broke 7
baseline successes. **There is no direction here.** At 1 of 32 layers it is also
only 1.03x faster, which is what 3% of the stack should buy. Nothing about this
cell supports "training-free pruning is a free lunch on OpenVLA"; it supports
"one layer changes which episodes succeed without changing how many".

Action repeat at p=0.057 is suggestive but not established on its own. What
raises confidence is that the same flag on the same backbone cost **−8.0** on
LIBERO: two benchmarks, one intervention, −8.3 and −8.0.

## This contradicts the horizon-only reading

`UniVLA_Bridge_ActionRepeat.md` concluded that the collapse there was about
open-loop horizon rather than architecture. This run does not fit that:

| backbone | baseline horizon | under repeat 2 | Δ |
|---|---|---|---|
| SpatialVLA | 1 | 2 | **+10.4** |
| **OpenVLA** | **1** | **2** | **−8.3** |
| UniVLA | 5 | 10 | −70.8 |

**Identical horizon change, opposite sign.** Horizon alone cannot be the
explanation.

### A hypothesis that fits all five measurements

Degradation tracks the distance from the horizon the policy was *trained* to
execute, not the absolute horizon:

| backbone | trained chunk | baseline executes | repeat 2 gives | direction |
|---|---|---|---|---|
| SpatialVLA | ~4 | 1 (re-plans every step) | 2 | **toward** → +10.4 |
| OpenVLA | 1 (no chunk) | 1 (matched) | 2 | away → −8.3 |
| UniVLA (Bridge) | 5 | 5 (matched) | 10 | far away → −70.8 |
| OpenVLA (LIBERO) | 1 | 1 (matched) | 2 | away → −8.0 |
| UniVLA (LIBERO) | 10 | 10 (matched) | 20 | far away → −68.0 |

SpatialVLA is the only backbone whose baseline deployment sits *below* its
trained horizon, and it is the only one that gains. It is also the case where
chunk-exec k=2 (45.9%) and action repeat 2 (42.7%) came out indistinguishable
(z=0.45) — consistent with "getting back to a multi-step horizon is what helps",
irrespective of what fills the extra step.

**This is a hypothesis, not a result.** It rests on SpatialVLA's +10.4, which is
the one number here that has never been paired-tested (unpaired z=1.50). Running
`paired_test.py` on the SpatialVLA/Bridge JSONs is the prerequisite for saying
any of this in the paper.

## What is actually established so far

| measurement | test | status |
|---|---|---|
| UniVLA/Bridge 5→10, −70.8 | McNemar p=1.2e-19 | **conclusive** |
| UniVLA/LIBERO 10→20, −68.0 | z=−9.8 | **conclusive** |
| OpenVLA 1→2, −8.3 (Bridge) and −8.0 (LIBERO) | paired p=0.057; z=−0.88 | suggestive, replicated |
| SpatialVLA 1→2, +10.4 | unpaired z=1.50 | **not established** |
| OpenVLA depth 1 of 32, +2.1 | paired p=0.80 | no effect detected |

At n=96 an unpaired comparison cannot resolve less than ~13 points, so "not
established" here means *not detected*, not *shown to be zero*.

## Reproduce

```bash
python RetinaBased/PythonProject/simple_eval.py --task <task> --n-episodes 24 \
  --openvla-model-path openvla/openvla-7b --openvla-unnorm-key bridge_orig \
  --device cuda --model openvla                        # original policy
  #                --model openvla_chunk --action-repeat 2
  #                --model openvla_depth --depth-prune 1
```

`openvla_chunk` is action repeat, not chunk execution — OpenVLA emits one action
per forward, so there is no predicted chunk to truncate. The class is
`ActionRepeatOpenVLAInference`; the name is historical.

Depth pruning selected layer **23** of 32 on every task, the same layer each
time, with the eligible range 16..31 (back half only) and the ranking rule
imported from `adaptive_sparse_vla/depth_prune.py` so it is identical to the
rule used on the other backbones.

Per-episode JSONs: `results/openvla_bridge_0805/`.

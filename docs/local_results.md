# Local experiment results

Date: 28 August 2026. The rollout table uses the Bridge MiniVLA checkpoint on
the SimplerEnv WidowX suite, with 200 episodes per condition (50 per task,
seed 42). These are not base-OpenVLA LIBERO results. Execution time is the
average complete episode time among successful episodes; all-episode values are
included because unsuccessful episodes often run to the step limit.

## The 14 independent rollout configurations

| configuration | success | rate | successful steps | successful episode time | all steps | all episode time |
|---|---:|---:|---:|---:|---:|---:|
| `original` | 72/200 | 36.0% | 36.50 | 5.376 s | 63.84 | 9.542 s |
| `fixed_foveation_keep20` | 56/200 | 28.0% | 32.38 | 5.202 s | 63.37 | 10.152 s |
| `fixed_foveation_keep50` | 70/200 | 35.0% | 34.89 | 5.381 s | 62.91 | 9.839 s |
| `action_repeat2` | 60/200 | 30.0% | 40.13 | 3.964 s | 66.94 | 6.636 s |
| `action_repeat4` | 58/200 | 29.0% | 40.40 | 3.026 s | 66.92 | 4.956 s |
| `depth_pruning1` (layer 13) | 37/200 | 18.5% | 35.32 | 5.102 s | 67.44 | 9.842 s |
| `depth_pruning2` (layers 11,13) | 0/200 | 0.0% | N/A | N/A | 75.00 | 10.344 s |
| `depth_pruning4` (layers 7,9,11,13) | 0/200 | 0.0% | N/A | N/A | 75.00 | 9.843 s |
| `guarded_reuse_strict` | 77/200 | 38.5% | 37.27 | 5.438 s | 62.65 | 9.117 s |
| `guarded_reuse_moderate` | 72/200 | 36.0% | 36.22 | 5.367 s | 64.04 | 9.370 s |
| `guarded_reuse_aggressive` | 67/200 | 33.5% | 35.73 | 5.228 s | 65.37 | 9.468 s |
| `temporal_fusion_motion_entropy` | 78/200 | 39.0% | 32.99 | 4.865 s | 61.17 | 9.167 s |
| `temporal_fusion_task_aware` | 71/200 | 35.5% | 34.76 | 5.550 s | 63.64 | 10.309 s |
| `temporal_fusion_conservative_adaptive` | 77/200 | 38.5% | 36.08 | 5.269 s | 63.09 | 9.375 s |

Each configuration has its own `summary.json` and `episodes.jsonl` below
[`artifacts/results/minivla_simplerenv/`](../artifacts/results/minivla_simplerenv/).
The former “Guarded reuse, default” is now `guarded_reuse_strict`; the former
“Temporal fusion, current” is now `temporal_fusion_motion_entropy`.

### Temporal-fusion configurations

| configuration | keyframe interval | reuse budget | task relevance | median reusable tokens |
|---|---:|---:|---|---:|
| `temporal_fusion_motion_entropy` | 3 | 50% | disabled | 107 |
| `temporal_fusion_task_aware` | 3 | 50% | enabled | 68 |
| `temporal_fusion_conservative_adaptive` | 2 | 25% | disabled | 64 |

Fusion still runs the full model and does not skip policy calls. It is therefore
a representation/temporal-consistency experiment, not a measured acceleration
result. Selective KV/KQV reuse remains a separate unintegrated speed path.

## Four-episode validation runs

These are smoke/validation checks, not statistically meaningful estimates.
They are stored as `validation_*` directories under the same model directory.

| configuration | success | rate | successful steps | successful episode time |
|---|---:|---:|---:|---:|
| `validation_original` | 2/4 | 50.0% | 29.00 | 4.371 s |
| `validation_fixed_foveation_keep20` | 3/4 | 75.0% | 24.00 | 3.774 s |
| `validation_action_repeat2` | 2/4 | 50.0% | 38.00 | 3.864 s |
| `validation_depth_pruning1` | 1/4 | 25.0% | 51.00 | 7.641 s |
| `validation_guarded_reuse_strict` | 2/4 | 50.0% | 29.00 | 4.399 s |
| `validation_temporal_fusion_motion_entropy` | 3/4 | 75.0% | 25.67 | 3.851 s |

## Base OpenVLA synthetic diagnostics

These measurements use optimized dense SDPA on RTX 5090 and establish
implementation behavior/latency only, not robot-task accuracy. They are stored
under [`artifacts/results/openvla_7b/`](../artifacts/results/openvla_7b/).

| diagnostic | median latency | speedup | observation |
|---|---:|---:|---|
| dense SDPA | 157.63 ms | 1.000x | reference |
| fixed foveation | 157.19 ms | ~1.000x | L2 change 0.0093; no model-speed mechanism |
| temporal fusion | 158.36 ms | ~1.000x | exact action; 128/256 tokens reusable |
| depth removal, layer 23 | 152.79 ms | 1.032x | L2 change 0.0089; gripper unchanged |
| depth removal, layers 23,26 | 148.11 ms | 1.057x | gripper flipped |
| guarded action reuse | ~0.2 ms reused call | — | two dense calls followed by one reuse |

The VLA-Cache sweep requested 50, 55, 65, 80, and 130 static patches. It
actually reused 16, 19, 26, 39, and 76 patches, respectively, with speedups
of 0.972x, 0.977x, 1.018x, 1.022x, and 1.025x. The first two preserved the
dense action; the last three flipped the gripper. This is retained in
`openvla_7b/vla_cache_sweep.json`.

## Claim boundary

The rollout numbers above are MiniVLA SimplerEnv evidence. They do not validate
base-OpenVLA LIBERO task success. Paper claims require suite-specific
base-OpenVLA checkpoints, disjoint calibration/validation/test splits, paired
seeds, and confidence intervals as specified in
[`docs/experiment_protocol.md`](experiment_protocol.md).

All data is in hand; nothing further to request. Final report follows.

## Inventory

- Files opened: **699** `episodes.jsonl`; records: **47,500**; malformed lines: 0; duplicate `(backbone, config, task, episode_index)` keys: 0.
- 13 backbone dirs × 14 configs = 182 runs. Layout differs: `cogact_base_simplerenv_bridge` and `minivla_simplerenv` have one config-level file (200 records); other SimplerEnv dirs have per-task files (50); LIBERO dirs have per-suite files (100 = 10 fine tasks × 10 episodes). `openvla_libero` has 55 files, not 56: `temporal_fusion_conservative_adaptive/libero_10` is missing (5,500 records vs 5,600).
- Six record schemas. `cronusvla_*` (6,300 records) lack `model_name`, `config_name`, `condition`, `seed`, `truncated`, `task_index`, `query_latency_ms`, `inference_cycle_ms`, `error_traceback` and carry `temporal_fused_patches`. `cogact_*` lack `inference_cycle_ms`. `smolvla_libero` lacks `truncated`/`task_index` and is split into a plain schema (2,895) and a SLURM-annotated schema (2,705: `gpu`, `qos`, `slurm_job_id`, `selected_depth_layers`, `recorded_at_utc` 2026-09-06/07). `spatialvla_*` `model_name` is `spatialvla_4b_simplerenv_*` (differs from dir name).

CSV: `/tmp/claude-0/-home-user-BiVLA/4327cefa-ba5d-5e2e-9d87-c73dd3a8ccbb/scratchpad/per_task_success.csv` — 2,286 rows (LIBERO at fine-task granularity; `task_group` column = suite/task dir), with n, successes, success_rate, mean_steps, truncation_rate (flag-based, null where absent), mean_policy_calls, mean_reuses, mean_episode_elapsed_s, median_of_episode_query_median_ms. Scripts: `load.py`, `analyze.py`, `followup.py` in the same directory.

Overall success (%, n in parentheses in CSV):

| backbone | orig | ar2 | ar4 | dp1 | dp2 | dp4 | fov20 | fov50 | gr_s | gr_m | gr_a | tf_ca | tf_me | tf_ta |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cogact bridge | 50 | 12 | 4 | 57 | 60 | 52 | 52 | 52 | 52 | 50 | 53 | 52 | 58 | 58 |
| cogact fractal | 67 | 66 | 48 | 65 | 66 | 67 | 65 | 67 | 67 | 67 | 67 | 66 | 65 | 65 |
| cronusvla bridge | 34 | 4 | 0 | 36 | 36 | 28 | 36 | 34 | 36 | 36 | 36 | 34 | 34 | 34 |
| cronusvla fractal | 56 | 58 | 46 | 56 | 52 | 51 | 54 | 57 | 54 | 55 | 57 | 57 | 57 | 57 |
| minivla | 36 | 30 | 29 | 18 | 0 | 0 | 28 | 35 | 38 | 36 | 34 | 38 | 39 | 36 |
| openvla libero | 74 | 68 | 50 | 68 | 60 | 45 | 56 | 72 | 72 | 69 | 71 | 79* | 73 | 71 |
| openvla bridge | 44 | 28 | 14 | 44 | 48 | 38 | 22 | 30 | 46 | 42 | 41 | 48 | 41 | 50 |
| openvla fractal | 36 | 40 | 36 | 38 | 41 | 37 | 8 | 24 | 36 | 36 | 35 | 37 | 42 | 42 |
| smolvla libero | 73 | 64 | 46 | 52 | 34 | 18 | 54 | 61 | 71 | 70 | 73 | 72 | 73 | 71 |
| spatialvla bridge | 45 | 41 | 29 | 38 | 28 | 14 | 50 | 44 | 45 | 45 | 46 | 45 | 46 | 44 |
| spatialvla fractal | 59 | 59 | 30 | 54 | 57 | 43 | 55 | 57 | 59 | 60 | 59 | 59 | 58 | 60 |
| univla libero | 92 | 27 | 0 | 94 | 92 | 89 | 91 | 96 | 92 | 92 | 92 | 92 | 92 | 92 |
| univla bridge | 88 | 12 | 3 | 82 | 82 | 82 | 79 | 73 | 88 | 88 | 86 | 88 | 82 | 48** |

\* 300 episodes, libero_10 missing; on the same 3 suites original = 80.0%, tf_ca = 79.3%. \*\* 81/200 episodes are CUDA OOM errors (see Q4).

## Q1. Per-task success, original vs configuration (successes/n)

action_repeat on WidowX (caps 60; eggplant 120):

| backbone | task | original | ar2 | ar4 |
|---|---|---|---|---|
| univla bridge | carrot_on_plate | 42/50 (84%) | 12/50 (24%) | 2/50 (4%) |
| | put_eggplant_in_basket | 49/50 (98%) | 3/50 (6%) | 1/50 (2%) |
| | spoon_on_towel | 44/50 (88%) | 6/50 (12%) | 3/50 (6%) |
| | stack_cube | 40/50 (80%) | 4/50 (8%) | 0/50 (0%) |
| cogact bridge | carrot_on_plate | 19/50 (38%) | 7/50 (14%) | 4/50 (8%) |
| | put_eggplant_in_basket | 34/50 (68%) | 8/50 (16%) | 1/50 (2%) |
| | spoon_on_towel | 31/50 (62%) | 9/50 (18%) | 3/50 (6%) |
| | stack_cube | 16/50 (32%) | 0/50 (0%) | 1/50 (2%) |
| cronusvla bridge | carrot_on_plate | 8/50 (16%) | 2/50 (4%) | 0/50 |
| | put_eggplant_in_basket | 48/50 (96%) | 4/50 (8%) | 0/50 |
| | spoon_on_towel | 12/50 (24%) | 1/50 (2%) | 0/50 |
| | stack_cube | 0/50 (0%) | 0/50 | 0/50 |
| openvla bridge | carrot_on_plate | 16/50 (32%) | 10/50 (20%) | 11/50 (22%) |
| | put_eggplant_in_basket | 37/50 (74%) | 23/50 (46%) | 14/50 (28%) |
| | spoon_on_towel | 19/50 (38%) | 14/50 (28%) | 3/50 (6%) |
| | stack_cube | 15/50 (30%) | 9/50 (18%) | 1/50 (2%) |

depth_pruning:

| backbone | task | original | dp1 | dp2 | dp4 |
|---|---|---|---|---|---|
| minivla | carrot_on_plate | 2/50 (4%) | 0/50 | 0/50 | 0/50 |
| | put_eggplant_in_basket | 9/50 (18%) | 10/50 (20%) | 0/50 | 0/50 |
| | spoon_on_towel | 25/50 (50%) | 19/50 (38%) | 0/50 | 0/50 |
| | stack_cube | 36/50 (72%) | 8/50 (16%) | 0/50 | 0/50 |
| spatialvla bridge | carrot_on_plate | 13/50 (26%) | 10/50 (20%) | 8/50 (16%) | 13/50 (26%) |
| | put_eggplant_in_basket | 50/50 (100%) | 46/50 (92%) | 41/50 (82%) | 8/50 (16%) |
| | spoon_on_towel | 11/50 (22%) | 9/50 (18%) | 0/50 (0%) | 3/50 (6%) |
| | stack_cube | 16/50 (32%) | 12/50 (24%) | 7/50 (14%) | 4/50 (8%) |
| smolvla libero | libero_10 | 42/100 | 9/100 | 3/100 | 0/100 |
| | libero_goal | 80/100 | 62/100 | 53/100 | 28/100 |
| | libero_object | 94/100 | 73/100 | 40/100 | 13/100 |
| | libero_spatial | 76/100 | 62/100 | 38/100 | 29/100 |

fixed_foveation on OpenVLA:

| env | task | original | keep20 | keep50 |
|---|---|---|---|---|
| bridge | carrot_on_plate | 16/50 (32%) | 12/50 (24%) | 17/50 (34%) |
| | put_eggplant_in_basket | 37/50 (74%) | 8/50 (16%) | 12/50 (24%) |
| | spoon_on_towel | 19/50 (38%) | 19/50 (38%) | 24/50 (48%) |
| | stack_cube | 15/50 (30%) | 6/50 (12%) | 7/50 (14%) |
| fractal | close_drawer | 20/50 (40%) | 15/50 (30%) | 17/50 (34%) |
| | move_near | 31/50 (62%) | 3/50 (6%) | 16/50 (32%) |
| | open_drawer | 9/50 (18%) | 0/50 (0%) | 5/50 (10%) |
| | pick_coke_can | 29/50 (58%) | 1/50 (2%) | 23/50 (46%) |
| | place_apple_in_closed_top_drawer | 0/50 | 0/50 | 0/50 |
| libero | libero_10 | 56/100 | 30/100 | 44/100 |
| | libero_goal | 75/100 | 64/100 | 75/100 |
| | libero_object | 87/100 | 74/100 | 86/100 |
| | libero_spatial | 78/100 | 55/100 | 83/100 |

## Q2. Concentrated vs spread

- Spread (every task drops by a large margin): univla bridge ar2/ar4 (all four tasks −60 to −92pp; stack_cube 0% at ar4); univla libero ar4 (all suites ≤1%; 38 of the 40 fine tasks with original ≥50% are at 0%); univla libero ar2 (all suites 21–34%; 7/40 fine tasks at 0%); cogact bridge ar2/ar4 (stack_cube 0% at ar2, the rest 2–18%); cronusvla bridge ar4 (all 0%; stack_cube was already 0% in original); minivla dp2/dp4 (4/4 tasks at 0%); smolvla dp4 (all suites, 20/40 fine tasks at 0%, 18 of them from ≥50%).
- Concentrated in one task: minivla dp1 (−18pp overall) is stack_cube 72%→16% (−56pp) while eggplant 18→20 and spoon 50→38; spatialvla bridge dp4 (−31pp) is eggplant 100%→16% (−84pp) with carrot unchanged 26→26; spatialvla dp2 (−17pp) has spoon 22→0; openvla bridge fov20 (−21pp) is eggplant 74→16 with spoon unchanged 38→38; openvla fov50 bridge likewise eggplant 74→24; openvla fractal fov20 is move_near 62→6, pick_coke 58→2, open_drawer 18→0 while close_drawer only 40→30; openvla bridge ar4: carrot barely moves (32→22) while spoon 38→6 and stack 30→2.
- smolvla depth pruning: libero_10 is hit hardest at every level (42→9→3→0); dp4 has 20/40 fine tasks at 0% but libero_goal task_4/task_7 stay 10/10 and spatial task_0/task_2 stay 7/10.
- univla bridge tf_task_aware 88→48% is not a model effect: 81 episodes errored (OOM); excluding them, 96/119 = 80.7%, and on the identical non-error episodes original scored carrot 42 vs 36, eggplant 7 vs 8, spoon 44 vs 44, stack 9 vs 8.

## Q3. Truncation

Caps (max steps observed, identical across all backbones on that env): WidowX bridge 60 for carrot_on_plate/spoon_on_towel/stack_cube and 120 for put_eggplant_in_basket; Google-robot fractal 80 for all 5 tasks; LIBERO libero_10 = 520, libero_goal = 300, libero_object = 280, libero_spatial = 220.

The `truncated` flag is unreliable and its meaning differs by directory:
- Absent in cronusvla (6,300) and smolvla (5,600).
- Bridge dirs (cogact, minivla, openvla, spatialvla, univla): flag == (steps == per-task cap) exactly, including 18 successes with `truncated=true` at cap (5 cogact, 3 minivla, 5 openvla bridge, 2 openvla fractal, 2 spatialvla, 1 univla).
- Fractal dirs (cogact/openvla/spatialvla): flag is set only for move_near and pick_coke_can at cap; close_drawer, open_drawer, place_apple at 80 steps carry `truncated=false` (cogact 1,122, openvla 1,662, spatialvla 1,361 such failures).
- LIBERO dirs (openvla, univla): flag is always false, including 1,872 (openvla) and 1,062 (univla) failures at cap.

Using steps == per-task cap: **every failure in the dataset except the 81 OOM episodes is a step-cap timeout** — 181 of 182 runs have 100% of failures at cap; the exception is univla bridge tf_task_aware (23/104). Failures never terminate early otherwise (0 non-cap failures in the other 12 backbones). Step-cap hit rate (all episodes) by run:

| backbone | orig | ar2 | ar4 | dp1 | dp2 | dp4 | fov20 | fov50 | gr_s | gr_m | gr_a | tf_ca | tf_me | tf_ta |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cogact bridge | .50 | .89 | .95 | .43 | .41 | .47 | .47 | .47 | .48 | .51 | .47 | .48 | .42 | .42 |
| cogact fractal | .34 | .34 | .52 | .35 | .34 | .33 | .35 | .33 | .34 | .33 | .33 | .35 | .35 | .35 |
| cronusvla bridge | .67 | .96 | 1.00 | .65 | .65 | .71 | .64 | .66 | .64 | .64 | .64 | .67 | .67 | .67 |
| cronusvla fractal | .47 | .47 | .84 | .44 | .48 | .49 | .47 | .44 | .47 | .47 | .44 | .44 | .44 | .44 |
| minivla | .64 | .70 | .71 | .81 | 1.00 | 1.00 | .72 | .66 | .61 | .64 | .67 | .61 | .61 | .65 |
| openvla libero | .26 | .33 | .51 | .32 | .40 | .55 | .44 | .28 | .28 | .31 | .29 | .21 | .28 | .29 |
| openvla bridge | .56 | .73 | .85 | .56 | .53 | .62 | .78 | .70 | .55 | .58 | .59 | .52 | .59 | .51 |
| openvla fractal | .64 | .60 | .64 | .62 | .59 | .63 | .92 | .76 | .64 | .64 | .65 | .64 | .58 | .58 |
| smolvla libero | .27 | .36 | .54 | .48 | .67 | .82 | .47 | .39 | .29 | .30 | .27 | .28 | .27 | .29 |
| spatialvla bridge | .55 | .59 | .71 | .61 | .72 | .86 | .51 | .56 | .55 | .55 | .55 | .55 | .54 | .56 |
| spatialvla fractal | .41 | .41 | .70 | .46 | .44 | .57 | .45 | .43 | .41 | .40 | .41 | .41 | .42 | .40 |
| univla libero | .08 | .73 | .99 | .06 | .09 | .11 | .09 | .04 | .07 | .07 | .08 | .08 | .08 | .08 |
| univla bridge | .12 | .88 | .97 | .18 | .18 | .18 | .21 | .28 | .12 | .12 | .14 | .12 | .18 | .12 |

Successes at exactly the cap: cronusvla fractal 121 (72 move_near, 47 pick_coke_can — that harness runs those tasks to the limit and scores at the end; other backbones have ≤5 such successes per dir).

## Q4. Errors

81 records with non-null `error`, all in `univla_simplerenv_bridge/temporal_fusion_task_aware`, all `OutOfMemoryError('CUDA out of memory. Tried to allocate 544.00 MiB ...' ` (42, put_eggplant_in_basket) / `'... 546.00 MiB ...'` (39, stack_cube); GPU 31.36 GiB, ~30.9 GiB used. Errored episodes: success=false, truncated=false, steps_executed 5 (63 of them), 15–55 (18), policy_calls 1–11, episode 0 never errors. 81 tracebacks present; 0 errors anywhere else.

## Q5. Latency

Original, per-episode `query_latency_ms.median_ms` (ms):

| backbone | n | min | p10 | median | p90 | max | elapsed/policy_call |
|---|---|---|---|---|---|---|---|
| cogact bridge | 200 | 97.8 | 98.1 | 98.7 | 99.4 | 100.0 | 139.8 |
| cogact fractal | 250 | 97.6 | 98.2 | 99.0 | 99.8 | 162.5 | 167.5 |
| cronusvla bridge/fractal | 0 | no field | | | | | 126.3 / 152.6 |
| minivla | 200 | 96.5 | 96.8 | 97.1 | 97.4 | 97.7 | 147.2 |
| openvla libero | 400 | 154.8 | 155.6 | 158.5 | 160.2 | 160.9 | 175.5 |
| openvla bridge | 200 | 157.4 | 158.3 | 159.0 | 159.6 | 161.5 | 208.3 |
| openvla fractal | 250 | 157.2 | 157.7 | 160.1 | 161.1 | 165.0 | 238.0 |
| smolvla libero | 400 | 269.0 | 270.5 | 275.1 | 280.2 | 285.2 | 292.3 |
| spatialvla bridge | 200 | 375.9 | 376.8 | 382.4 | 385.1 | 388.4 | 421.3 |
| spatialvla fractal | 250 | 372.2 | 375.3 | 379.0 | 381.3 | 385.5 | 446.9 |
| univla libero | 400 | 399.8 | 435.2 | 473.6 | 515.4 | 612.3 | 623.6 |
| univla bridge | 200 | 490.8 | 681.4 | 714.5 | 746.9 | 762.7 | 902.3 |

No run has zero latencies, repeated identical per-episode values (every run has n unique values), p95 < median, or any episode > 3× the run median (largest max/median = 1.64, cogact fractal original). cogact/minivla/openvla/spatialvla p10–p90 spans are 0.3–1.5% of the median; univla spans are 9–17%.

Ratio of run median `ql_median` to original:

| backbone | dp1 | dp2 | dp4 | tf_task_aware | tf_cons_adapt | tf_motion_ent | fov20 | fov50 |
|---|---|---|---|---|---|---|---|---|
| cogact bridge | 0.981 | 0.970 | 0.946 | 0.995 | 0.992 | 0.999 | 0.996 | 1.004 |
| cogact fractal | 0.991 | 0.979 | 0.951 | 0.998 | 0.994 | 1.021 | 1.002 | 1.003 |
| minivla | 0.986 | 0.944 | 0.878 | 1.136 | 1.006 | 1.015 | 1.027 | 1.010 |
| openvla libero | 0.965 | 0.943 | 0.884 | 1.076 | 1.003 | 0.998 | 0.997 | 0.999 |
| openvla bridge | 0.973 | 0.937 | 0.884 | 1.075 | 0.992 | 0.994 | 0.992 | 0.989 |
| openvla fractal | 0.973 | 0.943 | 0.891 | 1.080 | 0.993 | 1.004 | 1.039 | 1.005 |
| smolvla libero | 1.009 | 1.008 | 0.842 | 0.956 | 0.957 | 0.956 | 1.009 | 1.011 |
| spatialvla bridge | 0.963 | 0.933 | 0.874 | 1.013 | 1.005 | 1.006 | 1.005 | 1.001 |
| spatialvla fractal | 0.962 | 0.932 | 0.867 | 1.008 | 1.002 | 1.006 | 1.000 | 0.999 |
| univla libero | 0.976 | 0.957 | 0.903 | 1.128 | 0.992 | 0.994 | 1.001 | 0.996 |
| univla bridge | 0.973 | 0.954 | 0.905 | 1.205 | 1.007 | 1.002 | 0.999 | 1.001 |

- Depth pruning lowers per-call latency monotonically on 10 of 11 measurable backbones (dp4 = 0.87–0.95×). smolvla is confounded by hardware: its dp2/dp4/gr/tf runs are partially or wholly the SLURM schema on 5 GPU types. On the unlabeled hardware (same as original) smolvla dp4 = 273.3 ms vs 275.1 (0.99×) and dp2 = 277.9 (1.01×); the 0.842 comes from RTX 5090 episodes (229 ms). On RTX 5090 alone, task_aware 263.2 vs motion_entropy 262.5 vs guarded_reuse 261.5 vs conservative 263.0 — no task-aware overhead visible.
- Task-aware fusion raises per-call latency on univla bridge (+20.5%), minivla (+13.6%), univla libero (+12.8%), openvla (+7.5–8.0%); not on cogact (−0.5/−0.2%) or spatialvla (+0.8–1.3%). Conservative-adaptive and motion-entropy are within ±2% everywhere.
- cronusvla has no latency fields; `episode_elapsed_ms/policy_calls` shows dp4 0.90–0.93× and task_aware 1.01× of original.
- inference_cycle under action_repeat is per env step for minivla/openvla/smolvla (ar4: 2.3–13.4 ms) but per policy call for spatialvla/univla (ar4: 538–1514 ms) — the field means different things per backbone.

## Q6. Reuses

| backbone | gr_strict max/mean/zero-share | gr_moderate | gr_aggressive | ar2 max/mean/zero | ar4 max/mean/zero |
|---|---|---|---|---|---|
| cogact bridge | 13 / 0.16 / 0.94 | 20 / 0.48 / 0.87 | 45 / 1.35 / 0.80 | 0 / 0 / 1.00 | 0 / 0 / 1.00 |
| cogact fractal | 1 / 0.00 / 1.00 | 2 / 0.05 / 0.96 | 9 / 0.25 / 0.91 | 0 / 0 / 1.00 | 0 / 0 / 1.00 |
| cronusvla bridge | 1 / 0.03 / 0.97 | 1 / 0.03 / 0.97 | 1 / 0.03 / 0.97 | 60 / 36.7 / 0 | 90 / 56.3 / 0 |
| cronusvla fractal | 14 / 0.36 / 0.92 | 21 / 0.88 / 0.84 | 32 / 2.14 / 0.68 | 40 / 28.6 / 0 | 60 / 55.5 / 0 |
| minivla | 35 / 1.96 / 0.80 | 40 / 2.71 / 0.72 | 43 / 2.64 / 0.68 | 60 / 33.4 / 0 | 90 / 50.1 / 0 |
| openvla libero | 174 / 7.75 / 0.23 | 106 / 11.2 / 0.09 | 262 / 21.8 / 0.04 | 260 / 112 / 0 | 390 / 195 / 0 |
| openvla bridge | 23 / 1.84 / 0.74 | 30 / 2.51 / 0.66 | 28 / 3.31 / 0.64 | 60 / 30.2 / 0 | 90 / 51.2 / 0 |
| openvla fractal | 23 / 1.46 / 0.85 | 23 / 1.60 / 0.82 | 34 / 2.49 / 0.78 | 40 / 31.4 / 0 | 60 / 50.1 / 0 |
| smolvla libero | 9 / 0.27 / 0.88 | 17 / 0.91 / 0.68 | 77 / 4.97 / 0.33 | 260 / 115 / 0 | 390 / 198 / 0 |
| spatialvla bridge | 1 / 0.01 / 0.99 | 1 / 0.03 / 0.97 | 4 / 0.19 / 0.90 | 0 / 0 / 1.00 | 0 / 0 / 1.00 |
| spatialvla fractal | 25 / 0.26 / 0.93 | 26 / 0.41 / 0.82 | 34 / 1.20 / 0.66 | 0 / 0 / 1.00 | 0 / 0 / 1.00 |
| univla libero | 2 / 0.01 / 0.99 | 3 / 0.01 / 0.99 | 4 / 0.04 / 0.98 | 260 / 149 / 0 | 390 / 255 / 0 |
| univla bridge | 0 / 0.00 / 1.00 | 1 / 0.01 / 0.99 | 2 / 0.01 / 0.99 | 60 / 36.3 / 0 | 90 / 56.1 / 0 |

- Guarded reuse essentially never fires on: univla bridge (0/1/1 episodes with any reuse out of 200; reuse share 0.000), univla libero (3/3/8 of 400), cronusvla bridge (5/5/5 of 200 — and all 200 episodes are identical in (steps, success, reuses) across the three presets, so the presets are degenerate there), spatialvla bridge (2/6/20 of 200), cogact fractal strict (1/250). Only openvla libero reaches a meaningful reuse share (2.9–9.2% of steps); everything else is ≤5%.
- `reuses` is 0 for every action_repeat episode of cogact and spatialvla (their `policy_calls` still equal ceil(steps/k)); other backbones log reuses = steps − policy_calls (100%).
- Reuses are 0 in all original/dp/fov/tf episodes (no leakage).

## Q7. Seeds and pairing

- `seed` present in 41,200 records; null in 0; **absent in all 6,300 cronusvla records** (both dirs, all 14 configs). Where present, `seed == 42 + episode_index` in 41,200/41,200.
- `(task, episode_index)` set is identical across all 14 configs for 12 backbone dirs; the only difference is `openvla_libero/temporal_fusion_conservative_adaptive`, which lacks the 100 libero_10 pairs.
- 0 duplicate keys. `benchmark_episode_id` (spatialvla/univla) equals `episode_index` in 14,700/14,700. `task_index` is 0 for all tasks in openvla/spatialvla/univla SimplerEnv dirs (uninformative), 0–3 in cogact bridge/minivla, and the fine-task number (0–9) in LIBERO.

## Q8. Other anomalies

- steps_executed < policy_calls: 0 records. steps == 0 or policy_calls == 0: 0.
- action_repeat bookkeeping: `policy_calls == ceil(steps/k)` in 100% of episodes for every backbone except univla, which chunks actions: original steps/policy_calls fits chunk 10 on LIBERO and chunk 5 on bridge (100%), ar2 fits 20/10, ar4 fits 40/20 (100%). univla `reuses` = steps·(k−1)/k regardless (e.g. (60, 3, 45)); in non-repeat configs univla `policy_calls + reuses ≠ steps` in 100% of episodes.
- Wall-clock per env step under action_repeat is far from 1/k: ar4/original per-step time ratio is 0.28–0.31 (smolvla, openvla libero), 0.32–0.42 (spatialvla, univla, openvla bridge), 0.46–0.59 (cogact, cronusvla, minivla, openvla fractal); ar2 is 0.53–0.73. Per-policy-call elapsed rises under ar4 by +80–220 ms on cogact/cronusvla/openvla/spatialvla/univla (env stepping inside the call).
- LIBERO: every suite has exactly 10 fine tasks × 10 episodes in all three backbones; the only count deviation is the missing openvla libero_10 suite above. Fine-task caps equal suite caps (520/300/280/220).
- smolvla SLURM-schema subset: `selected_depth_layers` = [24,26,28,30] for the 262 dp4 records and [28,30] for the 43 dp2 records tagged (rest untagged); `fusion_reusable_tokens_median` = 0 for all guarded_reuse episodes, 16 for conservative_adaptive, 19–26 for motion_entropy, 2–7 for task_aware. Latency on RTX A6000 is 463–484 ms vs 261 on RTX 5090 and 184–189 on L40S; dp4 success by GPU spans 0% (L40S, n=18) to 42% (A6000, n=12).
- cronusvla `temporal_fused_patches` is 0 outside tf configs and takes per-episode values 1953/2031/6485/6584 in all three tf configs.
- Successes with `truncated=true`: 18 (all at cap, listed in `followup_out.txt` section B3).
- `condition` is null for all cronusvla records; elsewhere it maps cleanly (guarded_reuse_* → `guarded_action_reuse`).
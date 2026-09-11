## Scope and counts

Data root: `/home/user/BiVLA/artifacts/results/mentor_2026-09-11/`. Scripts: `/tmp/claude-0/-home-user-BiVLA/4327cefa-ba5d-5e2e-9d87-c73dd3a8ccbb/scratchpad/verify.py`, `verify2.py`, `verify3.py` (outputs `report.txt`, `report2.txt`).

| item | count |
|---|---|
| summary.json opened | 699 |
| episodes.jsonl opened | 699 (0 unparseable lines) |
| episode records | 47,500 |
| folders per backbone_env | cogact bridge 14, cogact fractal 70, cronusvla bridge 56, cronusvla fractal 70, minivla 14, openvla_libero 55, openvla_simplerenv 56, openvla fractal 70, smolvla 56, spatialvla bridge 56, spatialvla fractal 70, univla_libero 56, univla bridge 56 |
| CSVs | 25 (22 per-backbone, 307 rows: 140 SimplerEnv + 167 LIBERO; 3 summary.csv, 132 rows) |

Tolerances used: 0.01 on rates, 0.006 on means, 0.5 ms on latency medians. Latency assumption: per-call samples are not recorded, so `summary.median_ms` was compared to the median over episodes of the per-episode `median_ms` (and CSV medians to several derivations, stated below).

## LENS A.1 — summary.json vs episodes.jsonl (699 folders)

**Counts, rates, means: 0 mismatches.** `episodes`, `successes`, `success_rate`, `average_steps`, `average_policy_calls`, `average_reuses`, `average_episode_ms` and every `per_task.*` field agree with the recomputation in every folder where the field exists (SmolVLA `per_task` carries only episodes/successes; both agree).

**Status:** all 699 `status == "completed"`.

**Errors:** `episode_errors` is present in 377 folders (cogact, cronusvla, openvla_libero, smolvla, univla_libero), all 0; absent in 322 (openvla_simplerenv, openvla fractal, spatialvla ×2, univla bridge, minivla). Episode records with `error != null` exist in exactly two folders, both without the field:

| folder | error records | error text | counted as |
|---|---|---|---|
| `results/univla_simplerenv_bridge/temporal_fusion_task_aware/widowx_put_eggplant_in_basket` | 42 / 50 | `torch.OutOfMemoryError: CUDA out of memory` | failures (success=false, 5–25 steps, 1–5 calls); summary says episodes 50, successes 8, status completed |
| `results/univla_simplerenv_bridge/temporal_fusion_task_aware/widowx_stack_cube` | 39 / 50 | same | failures; summary episodes 50, successes 8 |

The 8 and 11 non-crashed episodes succeeded 8/8 and 8/11. These 81 OOM crashes propagate unflagged into `summarization/simpler_widowx/univla_simpler_widowx.csv` row `temporal_fusion_task_aware` (96/200 = 48.0%, vs original 175/200 = 87.5%), which is the Table I number.

**Latency medians (281 field-mismatches, all of them latency):**

| backbone_env | config group | folders | policy >0.5 ms | policy max dev | cycle >0.5 ms | cycle max dev |
|---|---|---|---|---|---|---|
| cogact bridge / fractal | all | 84 | 0 | 0.41 | 0 | 0.00 |
| minivla | action_repeat | 2 | 0 | 0.03 | 1 | 47.62 |
| minivla | other | 12 | 0 | 0.36 | 0 | 0.28 |
| openvla (libero, widowx, fractal) | action_repeat | 26 | 0 | 0.16 | 13 | 78.75 |
| openvla (3 envs) | other | 155 | 0 | 0.43 | 0 | 0.43 |
| smolvla | action_repeat | 8 | 0 | 0.38 | 5 | 135.40 |
| smolvla | other (19 with the field) | 19 | 0 | 0.38 | 0 | 0.45 |
| spatialvla bridge | all | 56 | 5 | 0.66 | 7 | 4.63 |
| spatialvla fractal | all | 70 | 9 | 1.75 | 51 | 9.52 |
| univla_libero | all | 56 | 46 | 14.71 | 52 | 19.71 |
| univla bridge | all | 56 | 41 | 52.44 | 51 | 81.28 |

Interpretation: the action-repeat cycle mismatches are structural — with k=2 the cycle latency is bimodal (OpenVLA ≈160 ms dense / ≈8 ms repeated); per-episode medians average the two modes (≈86 ms) while the summary reports ≈163 ms, i.e. the summary `median_ms` is a pooled per-call median, not a median of episode medians (with k=4 both give ≈7.8 ms). The same pooled-vs-nested effect explains the SpatialVLA/UniVLA deviations (within-episode latency variance is large there, e.g. UniVLA first-call cost). The two largest (52.4 / 81.3 ms) are the OOM folders above. Nothing here is recomputable exactly from the shipped files.

**Schema gaps:** CronusVLA (126 folders) records only episodes/successes/episode_errors/arguments — no rates, means, latency, per_task, fusion; its episodes carry no seed and no latency. SmolVLA: 29 folders (all guarded_reuse, all temporal_fusion, depth_pruning2/{libero_10,goal}, depth_pruning4/{libero_10,goal,object}) have no `arguments` and no latency blocks. SmolVLA runs mix hardware inside a folder (RTX 5090, RTX 6000 Ada, A6000, L40S, RTX PRO 6000 Blackwell), and three folders mix implementations: `depth_pruning2/libero_10` (83 legacy + 17 reconstructed), `depth_pruning2/libero_goal` (74 + 26), `depth_pruning4/libero_10` (38 + 62); success by implementation there: 3/83 vs 0/17, 41/74 vs 12/26, 0/38 vs 0/62.

## LENS A.2 — CSV row vs union of the configuration's folders (307 rows)

`episodes`, `successes`, `success_rate_pct`, `avg_steps`, `avg_policy_calls`, `avg_reuses`, `avg_episode_time_s`, `selected_depth_layers`, `fusion_keyframe_interval`, `fusion_max_reuse_fraction`: **0 mismatches** on 307/307 rows (the README's arrival check is confirmed). All 132 `summary.csv` rows are byte-identical to their backbone-CSV row.

**Latency medians — derivation that reproduces the CSV within 0.5 ms:**

| backbone | policy_median | cycle_median |
|---|---|---|
| cogact, minivla, openvla, smolvla, spatialvla | median of per-episode medians (all rows) | mostly median of episode medians; exceptions below |
| univla | 24 rows median-of-episode-medians, 46 rows mean-of-folder-medians (ambiguous within tol.) | libero: mean of folder medians; bridge: **none** (all 14 rows 6–16 ms below every candidate; e.g. `original` 902.345 vs 915.35/915.84/919.40) |
| cronusvla | not in episodes; equals median over episodes of `episode_elapsed_ms/policy_calls` (e.g. original 126.3, action_repeat2 162.782, fractal original 152.561) | median of `episode_elapsed_ms/steps_executed` (action_repeat2 81.45) |

Rows beyond 0.5 ms (53 total): the 14 UniVLA-bridge cycle rows above; OpenVLA cycle_median for action_repeat rows is derived differently per file — `simpler_widowx/openvla_simplerenv_widowx.csv` and `google_robot_fractal/openvla_simplerenv_fractal.csv` use the median over episodes of the per-episode cycle **mean** (47.479 / 47.82 vs summary median 7.8), whereas the four `libero/openvla_libero_*.csv` use the summary median (2.243); OpenVLA widowx guarded_reuse ×3 (0.5–1.0 ms), OpenVLA fractal keep20/keep50/task_aware (1.6–3.8 ms), SpatialVLA bridge ×4 (0.6–1.5 ms), SpatialVLA fractal ×9 (0.5–4.4 ms). The p95 columns are not reproducible by any single rule (mixed matches, 14–29 "none" per backbone).

**median_reusable_visual_tokens:** mean of folder medians for cogact/minivla/openvla/spatialvla/univla (47 rows). Mismatches: SmolVLA 12 rows read **0** in the CSV while episodes record `fusion_reusable_tokens_median` 19.5–23 (motion-entropy), 16 (conservative), 3–5 (task-aware) — see `libero/smolvla_libero_*.csv`. CronusVLA 6 rows (110.542 bridge, 115.285 fractal) are not reproducible from `temporal_fused_patches/policy_calls` (bridge mean 102.06, median 108.667, pooled 106.111; fractal 106.738/112.786/111.957).

**Setting columns:** `simpler_widowx/cronusvla_simpler_widowx.csv` and the fractal CronusVLA CSV give `fusion_keyframe_interval = 1` for all three fusion rows, yet episodes record ~102–118 fused patches per call, which cannot happen if every call is a keyframe; CronusVLA arguments contain no fusion keys. `minivla_simplerenv/temporal_fusion_motion_entropy/summary.json` arguments lack the fusion keys the CSV reports (3, 0.5); the recorded `keyframes` 4102 vs calls 12233 is consistent with N=3. CronusVLA CSV `action_repeat` column reads 1 / `depth_layers` 0 for non-repeat rows while its `arguments.action_repeat` is 2 (cosmetic).

## LENS B — Methods claims vs recorded fields

### 1. Depth pruning — holds for 6 backbones, violated for SmolVLA

Rule assumed (from `depth_calibration.rules` in every CronusVLA depth folder: `min_layer_fraction 0.25, protect_last 1, min_gap 1`): protected = indices `< floor(0.25·L)` and `L-1`; gap ≥ 2. Note the repo's own `adaptive_sparse_vla/depth_prune.py` defaults to `min_layer=0.5` with no last-layer protection; it is not what produced these files.

| backbone_env | L | dp1 | dp2 | dp4 | influence | rule |
|---|---|---|---|---|---|---|
| openvla_simplerenv / _fractal | 32 | [23] | [23,25] | [17,23,25,27] | yes | holds |
| openvla_libero (per suite 10/goal/object/spatial) | 32 | [23]/[17]/[17]/[8] | [23,25]/[17,20]/[17,19]/[8,20] | [19,21,23,25]/[8,17,20,23]/[17,19,21,23]/[8,20,23,25] | yes | holds (8 = floor(0.25·32), first unprotected index) |
| cogact bridge / fractal | 32 | [17]/[23] | [17,23] | [17,20,23,25]/[17,19,23,25] | yes | holds |
| spatialvla bridge / fractal | 26 | [8] | [8,10] | [8,10,13,23] | yes | holds |
| univla bridge | 32 | [26] | [26,30] | [21,24,26,30] | yes | holds |
| univla_libero (10/goal/object/spatial) | 32 | [19] | [19,21]/[19,21]/[19,21]/[19,22] | [19,21,26,30]/[19,21,26,29]/[19,21,26,29]/[19,22,26,29] | yes | holds |
| minivla | 24 | [13] | [11,13] | [7,9,11,13] | yes | holds |
| cronusvla bridge / fractal (DiT, 12) | 12 | [10] | [8,10] | [4,6,8,10]/[3,6,8,10] | yes | holds (3 = floor(0.25·12)) |
| **smolvla** | 16 | **[30]** | **[28,30]** | **[24,26,28,30]** | **none** (`calibrated: false` in 5 folders, no block in 7) | **violated: every index ≥ L in 12/12 folders** |

Block-Influence ranking: the greedy selector (ascending influence, protection, gap) reproduces the recorded set in **138/138** folders that record influence; sets are nested dp1 ⊂ dp2 ⊂ dp4 everywhere and identical across task folders of a WidowX/Fractal config. Leading protection was binding in 33 of 48 distinct selections (unprotected greedy would take layers 2–5); the gap rule in 40; last-layer protection was never binding, so it is not testable from the data. Boundary note: OpenVLA LIBERO uses layer 8 = floor(0.25·32); this holds under the floor rule and would fail an inclusive "first quarter" (0–8). SmolVLA corroboration: policy median latency does not fall with pruning (original 275.1 → dp1 277.6 → dp2 277.3 ms, dp4 231.8 on different GPUs) while every other backbone falls monotonically (OpenVLA widowx 159.0 → 154.6 → 149.0 → 140.6), yet SmolVLA success does change (libero_10 42 → 9, goal 80 → 62, object 94 → 73, spatial 76 → 62), so the indices refer to something other than the 16 kept layers; not resolvable from the files.

### 2. Temporal fusion — keyframe rule holds where recorded; conservative-adaptive is inert on SpatialVLA and UniVLA

Recorded settings (all backbones with arguments): motion-entropy N=3, fraction 0.5, event null, collect_relevance false; task-aware N=3, 0.5, collect_relevance true; conservative-adaptive N=2, 0.25, event threshold 0.03.

- N=3 folders (76 folders: cogact, minivla task-aware, openvla ×3, spatialvla ×2, univla ×2): `fusion.keyframes` equals Σ ceil(calls_i/3) **exactly in every folder** (ratio to calls/3 1.003–1.144). Holds.
- N=2 conservative-adaptive (37 folders): keyframes/Σ ceil(calls_i/2) = 1.50–2.00, i.e. the forced keyframe fires on 75–100% of calls. For SpatialVLA (9 folders) and UniVLA (8 folders) keyframes == calls and `median_reusable_visual_tokens = 0` — fusion never engages; UniVLA bridge conservative-adaptive is episode-identical to `original` (200/200 same success/steps/calls), UniVLA libero differs in 8/400 episodes, SpatialVLA fractal in 3/250. For OpenVLA/CogACT/MiniVLA the reusable-token median is exactly 64 = 0.25·256 (cap binding) and 128 = 0.5·256 for motion-entropy on LIBERO.
- Zero reusable tokens (18 folders): all 9 `spatialvla_*/temporal_fusion_conservative_adaptive/*`, all 8 `univla_*/temporal_fusion_conservative_adaptive/*`, and `univla_simplerenv_bridge/temporal_fusion_task_aware/widowx_put_eggplant_in_basket` (OOM).
- Not checkable: CronusVLA (no keyframes/N/fraction recorded; the three settings are episode-identical in success/steps/calls/fused patches on both envs, elapsed differs); SmolVLA (no arguments; only per-episode `fusion_reusable_tokens_median`, values 16 / 19.5–23 / 3–5 as above).

### 3. Guarded reuse — cap holds; settings not applied on CronusVLA bridge; SmolVLA unverifiable

Recorded gates: strict (frame 0.01, patch 0.03, cosine 0.995, cap 1), moderate (0.015, 0.04, 0.99, cap 1), aggressive (0.02, 0.05, 0.98, cap 2); translation floor 0.01 in all — consistent for cogact, openvla, spatialvla, univla, cronusvla fractal, minivla moderate/aggressive (`minivla_simplerenv/guarded_reuse_strict` arguments lack the reuse keys). **CronusVLA bridge**: all three settings record the strict defaults (cap 1) and are episode-identical (5 reuses in 9947 steps each). SmolVLA: no arguments in any guarded folder.

Cap: `reuses ≤ cap·calls` holds in **100%** of the 8,600 guarded-reuse episodes (0 violations). The bound as stated in the task, `reuses ≤ calls·cap/(cap+1)`, is exceeded in 27 episodes (e.g. `openvla_simplerenv/guarded_reuse_strict/widowx_carrot_on_plate` ep4: calls 39, reuses 21, cap 1; `openvla_libero/guarded_reuse_aggressive/libero_goal` ep4: calls 121, reuses 179, cap 2; full list in report2.txt §2) — but that bound does not follow from the cap: each dense call may be followed by up to `cap` reuses, so `reuses ≤ cap·calls`, equivalently `reuses/steps ≤ cap/(cap+1)`. All 27 are truncated failures. `steps == calls + reuses` holds in every non-UniVLA guarded episode.

Reuse rate (Σ reuses / Σ steps), strict / moderate / aggressive:

| backbone_env | strict | moderate | aggressive |
|---|---|---|---|
| openvla_libero | 0.036 | 0.052 | 0.101 |
| openvla_simplerenv | 0.037 | 0.048 | 0.063 |
| openvla fractal | 0.023 | 0.025 | 0.039 |
| minivla | 0.031 | 0.042 | 0.040 |
| cronusvla fractal | 0.007 | 0.017 | 0.042 |
| smolvla | 0.001 | 0.004 | 0.023 |
| cogact bridge | 0.003 | 0.009 | 0.027 |
| spatialvla fractal | 0.005 | 0.008 | 0.024 |
| cogact fractal | 0.0001 (1 reuse) | 0.001 | 0.006 |
| spatialvla bridge | 0.0002 (2) | 0.0007 (6) | 0.004 (38) |
| cronusvla bridge | 0.0005 (5) | 0.0005 (5) | 0.0005 (5) |
| univla_libero | 5 / 65,605 | 6 / 65,656 | 17 / 65,792 |
| univla bridge | 0 / 6,042 | 1 / 6,013 | 2 / 6,112 |

Nearly never reuse: UniVLA (both envs), SpatialVLA bridge, CronusVLA bridge, CogACT fractal strict.

### 4. Action repeat — holds

`policy_calls == ceil(steps_executed / (m·k))` in **100%** of episodes for all 13 backbone_envs and both k (2,600 episodes), with m=1 everywhere except UniVLA (m=5 bridge, m=10 libero, matching `checkpoint_manifest.native_action_chunk_size`; in `original` UniVLA has 0/200 and 0/400 episodes with steps==calls, max ceil(steps/calls)=5 and 10). SpatialVLA has native chunk 4 but executes one action per call (steps==calls in 450/450 original episodes). Accounting differs: CogACT and SpatialVLA record `reuses = 0` under action repeat, the others record `reuses = steps − calls` (UniVLA: `(k−1)·m·calls`); the CSV `avg_reuses` inherits this (0.0 for cogact/spatialvla action-repeat rows vs 30.165 for OpenVLA).

### 5. Foveation — holds

`arguments.fovea_keep_ratio` is 0.2 in all `fixed_foveation_keep20` folders and 0.5 in all `fixed_foveation_keep50` folders of all 13 backbone_envs (CronusVLA and SmolVLA included); the default 0.2 is echoed in every other folder's arguments, with `condition` distinguishing the applied control.

## README claims re-verified

seed == 42 + episode_index in every record carrying a seed (CronusVLA records none); episode sets identical across all 14 configurations per backbone_env, except `openvla_libero/temporal_fusion_conservative_adaptive` (300 episodes, libero_10 absent, so `libero/openvla_libero_10.csv` has 13 rows).
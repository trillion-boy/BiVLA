All checks are done; nothing further to request. Final report follows.

---

# Harness-settings audit of `mentor_2026-09-11` summary.json files

**Files opened: 699 / 699** (0 parse failures). `ROOT = /home/user/BiVLA/artifacts/results/mentor_2026-09-11/results/`; paths below are relative to ROOT.

**Index:** `/tmp/claude-0/-home-user-BiVLA/4327cefa-ba5d-5e2e-9d87-c73dd3a8ccbb/scratchpad/summary_json_index.csv` — 699 rows, 101 columns (path, backbone_dir, configuration, task_or_suite, status, episodes, successes, len_episode_errors, claim_scope, implementation_versions, policy_setup, every flattened `arguments.*`, `checkpoint_manifest.*`, `fusion.*`, `depth_calibration.*`). Same rows as JSONL alongside; full analysis dump in `analysis.txt`. (Scripts: `build_index.py`; my `analyze.py` was later overwritten in the shared scratchpad by another process — output preserved in `analysis.txt`.)

**Schema variants (explains most "absent" cells):**
- *Standard* (openvla×3, spatialvla×2, univla×2, minivla, cogact×2): `arguments`, `fusion`, `depth_calibration`, `claim_scope`, per_task, latency. `checkpoint_manifest` full on openvla/spatialvla/univla/minivla; on cogact_fractal only `{policy_setup, action_statistics_key}`; on cogact_bridge absent.
- *CronusVLA* (126 files): minimal — `status, config_name, condition, episodes, successes, episode_errors, arguments`. No model_name, per_task, latency, average_reuses, fusion, checkpoint_manifest. `depth_calibration` only in the 27 depth files. `claim_scope`/`task_suite_name`/`policy_setup`/`action_statistics_key` only on fractal; bridge has none of them.
- *SmolVLA* (56): two generations. 27 "legacy" (all 4 suites of original/action_repeat×2/fixed_foveation×2/depth_pruning1 + dp2 object,spatial + dp4 spatial): `arguments`, `checkpoint_manifest{policy,vlm,lerobot,transformers}`, `fusion: null`, `depth_calibration{selected_layers, calibrated:false}`. 29 "reconstructed" (guarded_reuse×3, temporal_fusion×3 all suites; dp2 libero_10,goal; dp4 libero_10,goal,object): `implementation_versions` list, **no `arguments`, no checkpoint_manifest, no fusion, no depth_calibration**; claim_scope "Reconstructed continuation. Legacy and new implementation/hardware timings are not a matched comparison; no improvement claim."

## 1. Attention backend and library versions

| backbone | checkpoint_manifest.attention_backend | arguments.attn_backend | implementation_versions | transformers | lerobot |
|---|---|---|---|---|---|
| spatialvla_simplerenv_bridge (56) / _fractal (70) | `native_mixed (SDPA where supported; ZoeDepth compatible)` | `sdpa` | – | – | – |
| univla_libero (56) / univla_simplerenv_bridge (56) | `sdpa` | `sdpa` | – | – | – |
| openvla_libero (55), openvla_simplerenv (56), openvla_simplerenv_fractal (70) | absent | absent | – | – | – |
| minivla, cogact×2, cronusvla×2 | absent | absent | – | – | – |
| smolvla_libero legacy (27) | absent | absent | – | `4.51.3 (isolated)` | `0.4.4` |
| smolvla_libero reconstructed (29) | absent | absent | `["smolvla-reconstructed-sdpa-v1"]` (26); `["legacy-original-source-unavailable","smolvla-reconstructed-sdpa-v1"]` (3: dp2/libero_10, dp2/libero_goal, dp4/libero_10) | – | – |

Example files: `spatialvla_simplerenv_bridge/original/widowx_carrot_on_plate/summary.json`, `univla_libero/original/libero_10/summary.json`. No file anywhere records a torch version. No SimplerEnv run other than SpatialVLA/UniVLA records a backend.

## 2. Relevance (text→vision attention) collection, temporal_fusion configs

| backbone | motion_entropy | task_aware | conservative_adaptive | evidence field |
|---|---|---|---|---|
| openvla_libero / _simplerenv / _fractal | False | **True** | False | `fusion.collect_relevance` = `arguments.fusion_collect_relevance` |
| spatialvla_bridge / _fractal | False | **True** | False | same |
| univla_libero / _simplerenv_bridge | False | **True** | False | same |
| minivla_simplerenv | False | **True** | False | same (motion_entropy file carries only this flag, no other fusion args) |
| cogact_base_bridge / _fractal | **not collected** | **not collected** | not collected | no `collect_relevance` field at all; `fusion.task_relevance_supported: false` in all 84 CogACT files incl. task_aware |
| cronusvla_bridge / _fractal | not recorded | not recorded | not recorded | no `fusion` section and no `fusion_*` argument in any of 126 files |
| smolvla_libero | not recorded | not recorded | not recorded | 12 tf files are reconstructed variant (no arguments/fusion) |

CogACT task_aware is per-file identical to motion_entropy in `fusion.keyframes`, `median_reusable_visual_tokens`, `average_policy_calls`, `successes` (bridge: 3200 / 108.0 / 116 successes; fractal: all 5 tasks). On every other backbone that records the flag, task_aware ≠ motion_entropy per file. CronusVLA's three tf configs give identical per-task successes on both benchmarks (bridge 67/200 each = 5/48/14/0; fractal 143/250 each = 25/50/18/50/0).

## 3. Fusion settings

Values where recorded (openvla×3, spatialvla×2, univla×2, cogact×2, minivla task_aware+conservative):

| config | keyframe_interval | max_reuse_fraction | event_motion_threshold | motion_threshold | entropy_protect | task_protect | protect_radius |
|---|---|---|---|---|---|---|---|
| motion_entropy, task_aware (and every non-fusion config, as defaults) | 3 | 0.5 | null | 0.01 | 0.15 | 0.2 | 1 |
| conservative_adaptive | **2** | **0.25** | **0.03** | 0.01 | 0.15 | 0.2 | 1 |

No backbone deviates in value. Deviations are absence only: **CronusVLA — none of the 7 fusion args in any of 126 files (keyframe interval not recorded)**; SmolVLA legacy 27 record only keyframe_interval 3 / max_reuse 0.5 / event null (no motion/protect fields), reconstructed 29 nothing; MiniVLA motion_entropy file has no fusion args.

`fusion.representation`: CogACT "CogACT projected visual tokens before Llama cognition-token generation"; SpatialVLA "SpatialVLA 16x16 projected visual tokens before Gemma decoding"; UniVLA "Emu3 VQ-code fusion before UniVLA language-model input"; OpenVLA/MiniVLA no field.

`fusion.median_reusable_visual_tokens` for conservative_adaptive: 64.0 on OpenVLA/CogACT/MiniVLA, **0.0 on SpatialVLA and UniVLA (both benchmarks)**; UniVLA bridge task_aware carrot_on_plate also 0.0. `average_reuses` is 0.0 for every temporal_fusion file. `fusion.keyframes` per file ranges e.g. openvla_libero task_aware 4624–13699, spatialvla_bridge task_aware 446–892, univla_bridge task_aware 66–130 (full list in analysis.txt Q3).

## 4. Guarded reuse presets (`reuse_max_frame_mae / max_local_patch_mae / min_action_cosine / min_translation_norm / max_consecutive`)

| backbone | strict | moderate | aggressive |
|---|---|---|---|
| openvla×3, spatialvla×2, univla×2, cogact×2, cronusvla_fractal | 0.01/0.03/0.995/0.01/1 | 0.015/0.04/0.99/0.01/1 | 0.02/0.05/0.98/0.01/2 |
| **cronusvla_simplerenv_bridge** | 0.01/0.03/0.995/0.01/1 | **0.01/0.03/0.995/0.01/1** | **0.01/0.03/0.995/0.01/1** |
| minivla_simplerenv | **absent** (no reuse args) | 0.015/0.04/0.99/0.01/1 | 0.02/0.05/0.98/0.01/2 |
| smolvla_libero | absent (reconstructed, no arguments) | absent | absent |

Values are identical across backbones per preset. Strict values are also the harness default on all 576 non-reuse files that carry the keys. **CronusVLA bridge: all three presets carry the strict values** — e.g. `cronusvla_simplerenv_bridge/guarded_reuse_aggressive/widowx_stack_cube/summary.json` lines 19–23 (`reuse_max_frame_mae: 0.01 … reuse_max_consecutive: 1`); per-task successes identical for all three (7/49/16/0 = 72/200). On fractal the presets differ and successes differ (136/137/142 of 250). SmolVLA reconstructed reuse files still report differing `average_reuses` (strict 0.03–0.70, moderate 0.32–2.23, aggressive 1.82–11.47) but record no thresholds. CronusVLA has no `average_reuses` field.

## 5. Depth pruning

| backbone | dp1 | dp2 | dp4 | n_layers (len influence) | calibration fields |
|---|---|---|---|---|---|
| cogact_bridge | [17] | [17,23] | [17,20,23,25] | 32 | influence only |
| cogact_fractal | [23] | [17,23] | [17,19,23,25] | 32 | influence only |
| **cronusvla_bridge** | [10] | [8,10] | [4,6,8,10] | **12** | `mode: protected_calibrated`, `seed: 10000`, `tasks`: all 4 WidowX, `rules: {min_layer_fraction: 0.25, protect_last: 1, min_gap: 1}`; args `depth_indices ""`, `depth_calibration_seed 10000`, `depth_calibration_tasks` |
| **cronusvla_fractal** | [10] | [8,10] | [3,6,8,10] | **12** | same, tasks = all 5 google_robot |
| minivla | [13] | [11,13] | [7,9,11,13] | 24 | influence only |
| openvla_libero | 10:[23] goal:[17] obj:[17] spat:[8] | 10:[23,25] goal:[17,20] obj:[17,19] spat:[8,20] | 10:[19,21,23,25] goal:[8,17,20,23] obj:[17,19,21,23] spat:[8,20,23,25] | 32 | per-suite influence (per-suite checkpoint) |
| openvla_simplerenv, _fractal | [23] | [23,25] | [17,23,25,27] | 32 | influence only |
| spatialvla_bridge, _fractal | [8] | [8,10] | [8,10,13,23] | 26 | influence only |
| univla_libero | [19] | [19,21] (×3), spat [19,22] | [19,21,26,30] (10); [19,21,26,29] (goal,obj); [19,22,26,29] (spat) | 32 | per-suite influence |
| univla_bridge | [26] | [26,30] | [21,24,26,30] | 32 | influence only |
| smolvla_libero | [30] | [28,30] (obj, spat only) | [24,26,28,30] (spat only) | – | `influence: null`, `calibrated: false`; other dp2/dp4 suites reconstructed → no depth_calibration |

`arguments.depth_layers` = 1 on every non-depth file that has arguments; CronusVLA non-depth files carry only `depth_indices: ""`.

**CronusVLA:** the fields never name the module (no "DiT"/"llm"/"blocks" key); it is a 12-entry influence vector with a dominant first entry (0.807 bridge / 0.802 fractal) and the lowest values at indices 10, 11, 9, 8, 7, 6. Whether these are DiT blocks cannot be read from the summaries. The 27 CronusVLA depth files also differ in provenance from the rest: `output_dir` `…/cronusvla_simplerenv_{bridge,fractal}_protected/…` and relative checkpoint `models/cronusvla_0.5B_bridge_rt_1/…`, versus `…_corrected/…` + `/home/shriram003/soumya/…` on the other 44 bridge files.

**Rules evidence:** explicit only in CronusVLA `rules` (first quarter protected, last layer protected, gap ≥1). For every other calibrated backbone the rule is not stated but all 120 selected sets satisfy min(selected) ≥ n/4, no adjacent indices, and never include layer n−1 — while the lowest-influence layers are in the protected first quarter (idx 2–6 on OpenVLA/CogACT/UniVLA), so the protection was binding. CronusVLA fractal dp4 selects index 3 = exactly 12·0.25 (boundary inclusive). SmolVLA's `[24,26,28,30]` is a fixed stride-2 tail pattern, not calibrated.

## 6. Other run settings per backbone

| backbone | seed | trials | max_steps | device | img_res / center_crop | chunk / ens.temp | action_statistics_key | policy_setup | episodes/file | extras |
|---|---|---|---|---|---|---|---|---|---|---|
| cogact_bridge | 42 | 50 | 150 | cuda:0 | – | – | – | – | 200 (4 tasks in one file) | cfg_scale 1.5, num_ddim_steps 10 |
| cogact_fractal | 42 | 50 | 80 | cuda:0 | – | – | fractal20220817_data (manifest) | google_robot (manifest) | 50 | cfg_scale 1.5, ddim 10 |
| cronusvla_bridge | 42 | `trials` 50 | – | – | – | – | – | – | 50 | no claim_scope, no task_suite_name |
| cronusvla_fractal | 42 | `trials` 50 | – | – | – | – | fractal20220817_data (top-level) | google_robot (top-level) | 50 | |
| minivla | 42 | 50 | 150 | cuda:0 | 224 / False | – | bridge_dataset | – | 200 | per-config output_dirs `minivla_simplerenv_*_50` |
| openvla_libero | 42 | 10 | 220 spatial / 280 object / 300 goal / 520 libero_10 | cuda:0 | 224 / False | – | = suite | – | 100 | resume true; libero_root |
| openvla_simplerenv | 42 | 50 | 150 | cuda:0 | 224 / False | – | bridge_orig | – | 50 | |
| openvla_fractal | 42 | 50 | 80 | cuda:0 | 224 / False | – | fractal20220817_data | google_robot | 50 | |
| smolvla legacy 27 | 42 | 10 | – | **cuda** | – | – | – | – | 100 | `suite`, `vlm_checkpoint`, resume true |
| smolvla reconstructed 29 | – | – | – | – | – | – | – | – | 100 | no arguments |
| spatialvla_bridge | 42 | 50 | 150 | cuda:0 | – | 4 / −0.8 | bridge_orig/1.0.0 | – | 50 | |
| spatialvla_fractal | 42 | 50 | 80 | cuda:0 | – | 4 / −0.8 | fractal20220817_data/0.1.0 | google_robot | 50 | |
| univla_libero | 42 | 10 | 220/280/300/520 | cuda:0 | – | 10 / – | libero | – | 100 | environment_action_format libero, normalizer_key libero, predict_action_frames 10, resume true |
| univla_bridge | 42 | 50 | 150 | cuda:0 | – | 5 / – | bridge_robot | – | 50 | |

`action_repeat` = 4 only on action_repeat4, else 2; `fovea_keep_ratio` = 0.5 only on fixed_foveation_keep50, else 0.2 (every backbone with arguments). claim_scope is constant per backbone except SmolVLA (two texts); `univla_libero` claim_scope reads "paired SimplerEnv WidowX rollouts. UniVLA native five-action chunks…" (copied from bridge; chunk there is 10); cronusvla_bridge has no claim_scope.

## 7. Status, errors, resume

- `status`: "completed" in all 699.
- `episode_errors`: int 0 in 377 files; key absent in 322 (openvla_simplerenv 56, openvla_fractal 70, spatialvla ×2 126, univla_bridge 56, minivla 14). Never a list → no error messages anywhere; no non-zero counts.
- `arguments.resume: true` on openvla_libero (55), univla_libero (56), smolvla legacy (27); absent elsewhere.
- Missing file: `openvla_libero/temporal_fusion_conservative_adaptive/` has only libero_goal/object/spatial (no libero_10) → 55 files not 56.

## Answers to the open questions

- **(a) CogACT task-aware attention collected?** No. No `collect_relevance` field; `fusion.task_relevance_supported: false` in all 84 CogACT files; task_aware is per-file identical to motion_entropy (keyframes, median tokens, policy calls, successes).
- **(b) CronusVLA fusion keyframe interval:** not recorded — no `fusion` section and no `fusion_*` argument in any of 126 CronusVLA files; the three fusion configs produced identical per-task results on both benchmarks.
- **(c) CronusVLA WidowX reuse presets:** strict, moderate and aggressive all record the strict thresholds (0.01/0.03/0.995/0.01/1) and identical per-task successes (72/200). Fractal presets are distinct.
- **(d) OpenVLA LIBERO attention backend:** not recorded (no `attention_backend`/`attn_backend` in any openvla_* file). Only SpatialVLA (`native_mixed…`/`sdpa`) and UniVLA (`sdpa`/`sdpa`) record one.
- **(e) Library versions:** only SmolVLA legacy 27 files: transformers `4.51.3 (isolated)`, lerobot `0.4.4`; SmolVLA reconstructed 29 carry only the tag `smolvla-reconstructed-sdpa-v1`. No torch version and no versions for the other 11 backbones.
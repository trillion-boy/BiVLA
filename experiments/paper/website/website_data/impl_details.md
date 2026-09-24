# Additional Implementation Details

This page records the implementation details that the paper (Sections III and IV-A) states only in summary form. Everything below is read from three kinds of source, and each fact names its source:

- **Trick modules and harnesses in the code repository** (`adaptive_sparse_vla/foveation.py`, `depth_prune.py`, `bypass_layer.py`, `eval.py` (UniVLA on SimplerEnv WidowX), `eval_libero.py`, `inference.py`, `inference_libero.py`, `inference_openvla_libero.py`). These hold the foveation transform, the depth-pruning rule and the action-repeat loop. The harness that produced the 14-configuration records for the other backbones (the one that writes the `arguments`, `fusion` and `depth_calibration` blocks) is not part of this repository; its settings are read from the records it wrote.
- **The per-configuration records** under `experiments/datas/plots/results_corrected/<backbone_env>/<config>/` (`summary.json`, `episodes.jsonl`; LIBERO backbones have one folder per suite, CronusVLA, OpenVLA, SpatialVLA and UniVLA on SimplerEnv one folder per task). 700 summary files, 47,600 episode records, matching the 22 backbone-environment pairs x 14 configurations of Section IV-A.
- **The paper sources and notes** (`experiments/paper/setup.tex`, `bagoftricks.tex`, `results.tex`, `AttentionBackend_mentor.md`, `DataAudit_2026-09-11.md`, `experiments/data_audit_2026-09-11/lens_settings.md`).

The internal reports from an earlier three-backbone study (`experiments/Report_EN.md`, `LIBERO_Report_EN.md`, `Overview_EN.md`) are used only for mechanisms; none of their numbers apply to the submitted paper.

Notation: "records" means the `summary.json` and `episodes.jsonl` files above; "in-repo harness" means `eval.py` / `eval_libero.py`.

---

## 1. Foveation

**Which transform.** The paper's Eq. (2) is the function `foveate_image_blur` (`adaptive_sparse_vla/foveation.py:83-144`). It is the geometry-preserving variant: no pixel is displaced, the image size and therefore the visual-token count are unchanged (docstring, `foveation.py:88-94`; `bagoftricks.tex:76-81`). A second, log-polar variant exists in the same file (`foveate_image_logpolar`, `foveation.py:49-80`) and is the in-repo CLI default (`eval.py:195`, `eval_libero.py:227`); see Discrepancies, item D2.

**Exact construction** (all line numbers in `foveation.py`):

| Quantity | Definition | Source |
|---|---|---|
| Keep ratio rho | 0.2 or 0.5 of the image area | records `arguments.fovea_keep_ratio` = 0.2 (`fixed_foveation_keep20`) or 0.5 (`fixed_foveation_keep50`), every backbone |
| Fovea centre | image centre (W/2, H/2) | line 110-111 (`center=None`); the records carry no centre argument, and the configuration name is `fixed_foveation` |
| Sharp-disc radius r_s | `r0 = sqrt(rho * H * W / pi)` | line 117 |
| Outer radius | distance from the centre to the **farthest image corner**, `max_radius = hypot(max(cx, W-cx), max(cy, H-cy))` | lines 118-120 |
| tau(x) | `t = clip((d(x) - r0) / (max_radius - r0), 0, 1)` with `d(x)` the Euclidean distance to the centre | lines 121-126 |
| G3 | `cv2.GaussianBlur(frame, (0, 0), sigmaX=3.0)` | line 129 |
| G9 | `cv2.GaussianBlur(frame, (0, 0), sigmaX=9.0)` | line 130 |
| Weights | `w9 = clip(2t - 1, 0, 1)`, `w3 = clip(2t, 0, 1) - w9`, `w0 = 1 - w3 - w9` | lines 132-134 |
| Blend | float32 mixture, rounded and clipped to uint8 | lines 136-139 |
| Fovea | pixels with `d <= r0` copied back from the input, so the disc is bit-identical | lines 142-143 |

The weights are piecewise linear in tau: for tau in [0, 0.5] the pixel is `(1 - 2t) * I + 2t * G3(I)`, for tau in [0.5, 1] it is `(2 - 2t) * G3(I) + (2t - 1) * G9(I)`; at tau = 0.5 the pixel is exactly G3, at tau = 1 exactly G9. The Gaussian kernel size is not set explicitly (`ksize=(0, 0)`), so OpenCV derives it from sigma; for 8-bit input that rule gives 2 * round(3 sigma) + 1, i.e. 19 px for sigma 3 and 55 px for sigma 9, with OpenCV's default reflect-101 border. `keep_ratio >= 1` returns the input unchanged and `keep_ratio <= 0` applies the strongest blur everywhere (lines 107-108, docstring lines 96-99). The paper writes that tau reaches one "at the image boundary"; in the code it reaches one only at the farthest corner (item D3).

**Where the observation is replaced.** In both in-repo harnesses the foveated frame is given only to the policy; the simulator keeps stepping on the raw frame.

- SimplerEnv (`eval.py:555-559`): `policy_image = apply_foveation(image, ...)` immediately before `model.step(policy_image, ...)`; `env.step` receives the policy's action and the next raw frame is fetched with `get_image` (`eval.py:596-603`). `apply_foveation` (`eval.py:265-270`) calls the transform at the camera's native resolution, before the policy's own resize.
- LIBERO (`eval_libero.py:600-616`): the agent view is read from `obs["agentview_image"]`, rotated 180 degrees to match the training convention (`eval_libero.py:151-158`), foveated (`policy_image = apply_foveation(...)`, line 608) and passed to `model.step`. The renderer resolution is 256 x 256 (`--camera-resolution`, `eval_libero.py:276-278`), so foveation is applied at 256 x 256 and the policy resizes afterwards (OpenVLA: JPEG round-trip at quality 95 followed by a 224 x 224 Lanczos resize inside the wrapper, `inference_openvla_libero.py:68-103, 223`).
- Foveation is applied at every step (`--foveate-phase always`, the default; `eval.py:197-201`, `eval_libero.py:240-244`). A "pregrasp" option exists but no record shows it.

**Wrist camera on LIBERO.** The in-repo LIBERO harness has `--foveate-views {agent, both}` with default `agent`: the wrist view (`robot0_eye_in_hand_image`) is passed to the policy unfoveated unless `both` is given, in which case it is foveated about its own image centre (`eval_libero.py:246-253, 605-613`). OpenVLA is single-view and ignores the wrist image (`inference_openvla_libero.py:220`); the UniVLA LIBERO checkpoint requires the wrist view (`inference_libero.py:403-409`). The records for the LIBERO foveation runs carry only `arguments.fovea_keep_ratio` (checked on `univla_libero`, `openvla_libero`, `smolvla_libero` `fixed_foveation_keep20/*/summary.json`); no view flag is recorded, so **whether the wrist view of UniVLA and SmolVLA was foveated in the reported runs is not recorded** (item D2).

**Cost.** The blur runs on the CPU. Timed on an analysis machine, not on the campaign hardware, `foveate_image_blur` at keep 20 % took 10.97 ms per 256 x 256 frame and 45.5 ms per 640 x 480 frame (`experiments/paper/WarpOverhead.md:15-27`). This cost is inside the per-step latency of the foveation rows (Section 6).

---

## 2. Action repeat

**Execution.** The policy is called once, and every action it returned is executed k times in a row before the next call. In the in-repo harnesses the repeat is `np.repeat` semantics on the chunk, `[a, b] -> [a, a, b, b]`, not tiling (`eval.py:576-583`; `eval_libero.py:640-641`, `np.repeat(action_chunk, k, axis=0)`). During the repeated steps the simulator still returns an observation after every step (`eval.py:596-603`), but the policy is not queried and the observation given to the next call is the latest one.

**Interaction with the UniVLA chunk.** UniVLA emits a chunk per call, five actions on WidowX (`checkpoint_manifest.native_action_chunk_size` = 5 in `univla_simplerenv_bridge/*/*/summary.json`) and ten on LIBERO (`arguments.predict_action_frames` = 10 and `native_action_chunk_size` = 10 in `univla_libero/*/*/summary.json`). Repeat is applied to each element of the chunk after any chunk truncation (`eval.py:573-583`, and the `--action-repeat` help text at `eval.py:176-185`), so the open-loop horizon per observation becomes k x chunk. The records confirm it (pooled `steps_executed` / `policy_calls` per episode from `episodes.jsonl`):

| Backbone / env | k | steps per episode | calls per episode | steps per call |
|---|---|---|---|---|
| UniVLA WidowX | 1 (original) | 30.21 | 6.38 | 4.7 |
| UniVLA WidowX | 2 | 71.93 | 7.25 | 9.9 |
| UniVLA WidowX | 4 | 74.52 | 3.74 | 19.9 |
| UniVLA LIBERO (4 suites pooled) | 1 | 164.27 | 16.84 | 9.8 |
| UniVLA LIBERO | 2 | 296.07 | 14.93 | 19.8 |
| UniVLA LIBERO | 4 | 329.33 | 8.48 | 38.8 |
| CogACT WidowX | 2 | 70.61 | 35.33 | 2.0 |
| OpenVLA WidowX | 2 | 60.49 | 30.33 | 2.0 |
| OpenVLA LIBERO | 2 | 224.66 | 112.50 | 2.0 |
| SmolVLA LIBERO | 2 | 231.22 | 115.77 | 2.0 |

For single-action policies the call count is ceil(steps / k).

**Step counting.** Every simulator step counts as one environment step, whether it followed a fresh call or a repeat (`eval.py:610`; `eval_libero.py:646`), and every call counts as one policy call (`eval.py:562`; `eval_libero.py:618`). The repeat loop stops at the episode's termination or step cap (`eval.py:611-612`; `eval_libero.py:647-655`), so repeat never carries an episode past its cap. The recorded maxima of `steps_executed` under `action_repeat2` and `action_repeat4` equal the caps of Section 6.

**The `reuses` field under action repeat is harness-specific and must not be compared across backbones.** CogACT and SpatialVLA record `reuses = 0` under action repeat; OpenVLA, MiniVLA, SmolVLA and CronusVLA record `steps - calls` (OpenVLA WidowX k = 2: 30.16 reuses = 60.49 - 30.33); UniVLA records `(k - 1) x chunk x calls` without truncation at the cap (WidowX k = 2: 36.25 = 7.25 x 5 x 1; LIBERO k = 4: 254.55 = 8.48 x 10 x 3, which exceeds the step count of capped episodes). Source: `episodes.jsonl` of the `action_repeat*` folders, pooled; also `DataAudit_2026-09-11.md`, finding R6. `policy_calls` and `steps_executed` are the comparable fields.

**What repeat saves.** The per-call model time is unchanged and the per-step wall-clock falls, but not to 1/k, because the simulator step is paid on every environment step (`results.tex:104-107`). The recorded `inference_cycle_ms` has a different meaning per harness under repeat (per call including the repeated steps on CogACT, SpatialVLA and UniVLA; per step on the others; `DataAudit_2026-09-11.md`, R1), which is why the paper reports only the per-step wall-clock.

---

## 3. Depth pruning

**Block Influence as computed.** `depth_prune.py:71-110` (`measure_redundancy_with_hooks`) registers a forward hook on every decoder layer, takes the layer's input hidden states (`args[0]`, or `kwargs["hidden_states"]`) and its output (`output[0]`), computes the cosine similarity along the hidden dimension for every token position, averages over all positions and the batch, and stores `1 - mean cos`. Only the **first call** of each layer is recorded (lines 76-80): that is the prefill over the whole prompt (image tokens plus text); the single-token decode calls are excluded. The measurement must run on the unpruned stack (lines 81-83). For UniVLA the same quantity is taken from one extra no-cache forward with `output_hidden_states=True`, pairing consecutive hidden states (`inference_libero.py:323-351`; on WidowX `inference.py:1158-1180`); for OpenVLA the hooks ride on the first real `predict_action` call, so calibration costs no extra pass and that one call runs unpruned (`inference_openvla_libero.py:252-264`). The token average therefore runs over every prompt position, not only the visual tokens.

**Calibration frames.** For CogACT, MiniVLA, OpenVLA, SpatialVLA and UniVLA the record holds one influence array per run (`summary.json` `depth_calibration.influence`, no seed field) measured on the first test frame of the run, whose step runs unpruned; the ranking is then fixed for the rest of the run (`setup.tex:47-52`; `depth_prune.py:246-265` sets `_calibrated` once, and `reset_episode`, lines 297-307, does not clear it in static mode). Within one environment the influence arrays are identical across the task folders of a backbone (checked on `depth_pruning2` for OpenVLA WidowX and Fractal, SpatialVLA WidowX, CogACT Fractal, UniVLA WidowX), so one ranking served all tasks of that environment. On LIBERO each suite is a separate run (OpenVLA: a separate checkpoint per suite, `checkpoint_manifest.action_statistics_key` = suite) and the selected layers differ per suite. CronusVLA is the exception: its records name a **separate calibration pass** with `depth_calibration.mode = "protected_calibrated"`, `seed = 10000`, `tasks` = the environment's full task list, and `arguments.depth_calibration_seed = 10000`, `depth_calibration_tasks` (`cronusvla_simplerenv_*/depth_pruning*/*/summary.json`). Its 12-entry influence vector is the 12-layer DiT action decoder (`setup.tex:23-27`); the first entry dominates (0.807 on WidowX, 0.802 on Fractal). Note that for the six non-CronusVLA backbones the calibration frame is the first observation of the evaluation run itself (item D1).

**Protection rules.** The records are consistent with one rule set, stated explicitly only in the CronusVLA records: `rules = {min_layer_fraction: 0.25, protect_last: 1, min_gap: 1}`, i.e. layers below n/4 are never removed (boundary inclusive: CronusVLA Fractal at four layers selects index 3 = 12 x 0.25, and OpenVLA Spatial selects index 8 = 32 x 0.25; SpatialVLA's lowest selection, 8 of 26, also lies inside the window), the final layer is never removed, and no two removed layers are adjacent. All 120 calibrated selections satisfy these three constraints (`lens_settings.md`, "Rules evidence"). The early-window protection was binding: the lowest-influence layers of every 32-layer stack are early ones (OpenVLA WidowX, six lowest: 3, 2, 5, 4, 23, 6; UniVLA WidowX: 2, 4, 3, 26, 30, 25; CogACT Fractal: 2, 3, 4, 5, 23, 17; read from `depth_calibration.influence`), and the removed layers are the lowest-influence ones outside the window. The gap rule is visible on SpatialVLA, whose three lowest are 8, 9, 10 and whose two-layer selection is [8, 10]. The in-repo `DepthPruner` implements the window as `floor(min_layer x n) .. n-1` with default `min_layer = 0.5`, the gap as `min_gap = 1` with a greedy pass that skips neighbours and a second pass that fills the budget allowing adjacency if needed (`depth_prune.py:219-244`), and it does **not** protect the final layer; see item D1.

**Bypass mechanism.** A removed block is replaced in the `ModuleList` by `BypassDecoderLayer` (`depth_prune.py:184-216`; `bypass_layer.py:17-62`), which returns the hidden states unchanged and, when a KV cache is in use, writes a zero placeholder of the right shape (`num_kv_heads x head_dim`) at its layer index so the cache stays contiguous (`bypass_layer.py:44-56`); the placeholder is never read. The block's attention and MLP are not executed on prefill or on any decode token. Weights are untouched and the original modules are restored on `restore()` (`depth_prune.py:175-182`). The attention shape is read from the nested text config when the outer config has none (`depth_prune.py:196-210`).

**Layers actually removed** (`summary.json` `depth_calibration.selected_layers`; n = length of `depth_calibration.influence`):

| Backbone / environment | n | 1 layer | 2 layers | 4 layers | Calibration |
|---|---|---|---|---|---|
| CogACT WidowX | 32 | [17] | [17, 23] | [17, 20, 23, 25] | first frame of run |
| CogACT Fractal | 32 | [23] | [17, 23] | [17, 19, 23, 25] | first frame of run |
| OpenVLA WidowX | 32 | [23] | [23, 25] | [17, 23, 25, 27] | first frame of run |
| OpenVLA Fractal | 32 | [23] | [23, 25] | [17, 23, 25, 27] | first frame of run |
| OpenVLA LIBERO Long | 32 | [23] | [23, 25] | [19, 21, 23, 25] | first frame, per suite |
| OpenVLA LIBERO Goal | 32 | [17] | [17, 20] | [8, 17, 20, 23] | first frame, per suite |
| OpenVLA LIBERO Object | 32 | [17] | [17, 19] | [17, 19, 21, 23] | first frame, per suite |
| OpenVLA LIBERO Spatial | 32 | [8] | [8, 20] | [8, 20, 23, 25] | first frame, per suite |
| SpatialVLA WidowX | 26 | [8] | [8, 10] | [8, 10, 13, 23] | first frame of run |
| SpatialVLA Fractal | 26 | [8] | [8, 10] | [8, 10, 13, 23] | first frame of run |
| CronusVLA WidowX (DiT) | 12 | [10] | [8, 10] | [4, 6, 8, 10] | separate pass, seed 10000, all 4 tasks |
| CronusVLA Fractal (DiT) | 12 | [10] | [8, 10] | [3, 6, 8, 10] | separate pass, seed 10000, all 5 tasks |
| UniVLA WidowX | 32 | [26] | [26, 30] | [21, 24, 26, 30] | first frame of run |
| UniVLA LIBERO Long | 32 | [19] | [19, 21] | [19, 21, 26, 30] | first frame, per suite |
| UniVLA LIBERO Goal | 32 | [19] | [19, 21] | [19, 21, 26, 29] | first frame, per suite |
| UniVLA LIBERO Object | 32 | [19] | [19, 21] | [19, 21, 26, 29] | first frame, per suite |
| UniVLA LIBERO Spatial | 32 | [19] | [19, 22] | [19, 22, 26, 29] | first frame, per suite |
| MiniVLA WidowX | 24 | [13] | [11, 13] | [7, 9, 11, 13] | first frame of run |
| SmolVLA LIBERO (all suites) | not recorded | [30] | [28, 30] | [24, 26, 28, 30] | none: `calibrated: false`, `influence: null` |

SmolVLA's layers are fixed indices, not chosen by Block Influence (`smolvla_libero/depth_pruning1/*/summary.json`: `depth_calibration = {selected_layers: [30], calibrated: false}`; likewise `[28, 30]` on Object and Spatial at two layers and `[24, 26, 28, 30]` on Spatial at four layers; `setup.tex:50-52`). The remaining SmolVLA depth folders (two layers on Long and Goal, four layers on Long, Goal and Object) were completed under the reconstructed implementation and have no `depth_calibration` block; their per-episode records carry `selected_depth_layers` = [28, 30] and [24, 26, 28, 30] on the reconstructed episodes and `null` on the legacy ones (`episodes.jsonl`; Long two layers: 83 legacy + 17 reconstructed episodes, Goal two layers: 74 + 26, Long four layers: 38 + 62, Goal and Object four layers: 100 reconstructed each). The SmolVLA decoder is stated as 32 SmolLM2 layers (`setup.tex:25`; `Setup_ko.md:73-78`).

---

## 4. Guarded reuse

**Gates.** The decision of Eq. (5) has six conjuncts. The record key that carries each threshold, and the values of the three presets (`summary.json` `arguments`, identical across every backbone that records them):

| Gate | Statistic | Record key | Strict | Moderate | Aggressive |
|---|---|---|---|---|---|
| global image change d_g <= tau_g | mean absolute difference between cheap signatures of consecutive observations | `reuse_max_frame_mae` | 0.01 | 0.015 | 0.02 |
| local patch change d_l <= tau_l | maximum over local patches of the same difference | `reuse_max_local_patch_mae` | 0.03 | 0.04 | 0.05 |
| action agreement c_t >= tau_a | cosine similarity of the two most recent model-inferred 6-D pose actions | `reuse_min_action_cosine` | 0.995 | 0.99 | 0.98 |
| translation floor v_t >= tau_v | translation norm of the candidate reused action | `reuse_min_translation_norm` | 0.01 | 0.01 | 0.01 |
| gripper unchanged | commanded gripper state equal to the previous one | (no threshold) | yes | yes | yes |
| reuse cap r_t < R_max | consecutive reuses so far | `reuse_max_consecutive` | 1 | 1 | 2 |

The strict values are also the harness defaults: every non-reuse record that carries the keys holds them (`lens_settings.md`, Section 4). The construction of the "signature" (its resolution, normalisation and patch grid) is neither in this repository nor in the records. When any gate fails the harness queries the policy in full and the freshly inferred action is executed; when all pass, the previous executed action is repeated for one step and the observation is examined again at the next step (Eq. (5); `bagoftricks.tex:184-187`). With R_max = 1 (strict, moderate) a reused step is always followed by a full call; with R_max = 2 at most two consecutive steps are reused. Every reused step is counted in `reuses` and every full call in `policy_calls`, so `policy_calls + reuses = steps_executed` on single-action policies (CogACT WidowX strict: 50.17 + 0.16 = 50.33).

**Gate activity** (pooled over the `episodes.jsonl` of each folder; "reused %" = total reuses / total steps; "episodes with a reuse" = episodes with at least one reused step; LIBERO per suite):

| Backbone / env | Preset | Episodes | Steps / ep | Calls / ep | Reuses / ep | Reused % | Episodes with a reuse | Success % |
|---|---|---|---|---|---|---|---|---|
| CogACT WidowX | strict | 200 | 50.33 | 50.17 | 0.16 | 0.32 | 11 | 51.5 |
| CogACT WidowX | moderate | 200 | 51.84 | 51.35 | 0.49 | 0.94 | 26 | 50.0 |
| CogACT WidowX | aggressive | 200 | 49.77 | 48.41 | 1.36 | 2.72 | 41 | 53.0 |
| CogACT Fractal | strict | 250 | 44.70 | 44.70 | 0.004 | 0.01 | 1 | 66.8 |
| CogACT Fractal | moderate | 250 | 44.43 | 44.38 | 0.05 | 0.11 | 11 | 67.2 |
| CogACT Fractal | aggressive | 250 | 43.63 | 43.38 | 0.25 | 0.57 | 22 | 66.8 |
| OpenVLA WidowX | strict | 200 | 49.69 | 47.84 | 1.84 | 3.70 | 51 | 45.5 |
| OpenVLA WidowX | moderate | 200 | 52.12 | 49.60 | 2.51 | 4.82 | 68 | 41.5 |
| OpenVLA WidowX | aggressive | 200 | 52.70 | 49.39 | 3.31 | 6.27 | 72 | 41.0 |
| OpenVLA Fractal | strict | 250 | 64.29 | 62.83 | 1.46 | 2.28 | 37 | 36.0 |
| OpenVLA Fractal | moderate | 250 | 64.20 | 62.59 | 1.60 | 2.50 | 45 | 36.4 |
| OpenVLA Fractal | aggressive | 250 | 64.25 | 61.76 | 2.49 | 3.88 | 56 | 35.2 |
| OpenVLA LIBERO Long | strict | 100 | 397.87 | 383.50 | 14.37 | 3.61 | 92 | 51.0 |
| OpenVLA LIBERO Long | moderate | 100 | 396.62 | 374.49 | 22.13 | 5.58 | 95 | 51.0 |
| OpenVLA LIBERO Long | aggressive | 100 | 399.28 | 358.81 | 40.47 | 10.14 | 97 | 50.0 |
| OpenVLA LIBERO Goal | strict | 100 | 164.75 | 156.84 | 7.91 | 4.80 | 69 | 71.0 |
| OpenVLA LIBERO Goal | moderate | 100 | 165.24 | 156.76 | 8.48 | 5.13 | 87 | 69.0 |
| OpenVLA LIBERO Goal | aggressive | 100 | 163.88 | 145.59 | 18.29 | 11.16 | 92 | 72.0 |
| OpenVLA LIBERO Object | strict | 100 | 165.41 | 159.32 | 6.09 | 3.68 | 86 | 85.0 |
| OpenVLA LIBERO Object | moderate | 100 | 171.23 | 161.72 | 9.51 | 5.55 | 97 | 80.0 |
| OpenVLA LIBERO Object | aggressive | 100 | 167.10 | 151.02 | 16.08 | 9.62 | 100 | 85.0 |
| OpenVLA LIBERO Spatial | strict | 100 | 129.59 | 126.97 | 2.62 | 2.02 | 63 | 82.0 |
| OpenVLA LIBERO Spatial | moderate | 100 | 135.84 | 131.05 | 4.79 | 3.53 | 85 | 76.0 |
| OpenVLA LIBERO Spatial | aggressive | 100 | 137.62 | 125.22 | 12.40 | 9.01 | 93 | 77.0 |
| SpatialVLA WidowX | strict | 200 | 45.15 | 45.13 | 0.01 | 0.02 | 2 | 45.0 |
| SpatialVLA WidowX | moderate | 200 | 45.11 | 45.08 | 0.03 | 0.07 | 6 | 45.0 |
| SpatialVLA WidowX | aggressive | 200 | 45.12 | 44.94 | 0.19 | 0.42 | 20 | 45.5 |
| SpatialVLA Fractal | strict | 250 | 49.61 | 49.36 | 0.26 | 0.52 | 17 | 59.2 |
| SpatialVLA Fractal | moderate | 250 | 49.44 | 49.02 | 0.41 | 0.83 | 44 | 60.0 |
| SpatialVLA Fractal | aggressive | 250 | 49.93 | 48.73 | 1.20 | 2.40 | 85 | 58.8 |
| CronusVLA WidowX | strict | 200 | 49.73 | 49.71 | 0.03 | 0.05 | 5 | 36.0 |
| CronusVLA WidowX | moderate (rerun) | 200 | 50.14 | 49.98 | 0.17 | 0.33 | 19 | 34.5 |
| CronusVLA WidowX | aggressive (rerun) | 200 | 49.68 | 48.99 | 0.69 | 1.38 | 38 | 36.5 |
| CronusVLA Fractal | strict | 250 | 51.93 | 51.57 | 0.36 | 0.69 | 20 | 54.4 |
| CronusVLA Fractal | moderate | 250 | 51.87 | 50.99 | 0.88 | 1.70 | 41 | 54.8 |
| CronusVLA Fractal | aggressive | 250 | 50.73 | 48.59 | 2.14 | 4.22 | 80 | 56.8 |
| UniVLA WidowX | strict | 200 | 30.21 | 6.38 | 0.00 | 0.00 | 0 | 87.5 |
| UniVLA WidowX | moderate | 200 | 30.07 | 6.34 | 0.005 | 0.02 | 1 | 87.5 |
| UniVLA WidowX | aggressive | 200 | 30.56 | 6.43 | 0.01 | 0.03 | 1 | 86.5 |
| UniVLA LIBERO Long | strict | 100 | 280.58 | 28.40 | 0.05 | 0.02 | 3 | 86.0 |
| UniVLA LIBERO Long | moderate | 100 | 280.89 | 28.42 | 0.06 | 0.02 | 3 | 86.0 |
| UniVLA LIBERO Long | aggressive | 100 | 281.23 | 28.34 | 0.17 | 0.06 | 8 | 86.0 |
| UniVLA LIBERO Goal | strict / moderate / aggressive | 100 each | 121.34 / 121.54 / 122.56 | 12.53 / 12.56 / 12.65 | 0 / 0 / 0 | 0 | 0 | 93 / 93 / 92 |
| UniVLA LIBERO Object | strict / moderate / aggressive | 100 each | 145.82 | 14.98 | 0 | 0 | 0 | 96 / 96 / 96 |
| UniVLA LIBERO Spatial | strict / moderate / aggressive | 100 each | 108.31 | 11.27 | 0 | 0 | 0 | 95 / 95 / 95 |
| MiniVLA WidowX | strict | 200 | 62.65 | 60.69 | 1.96 | 3.13 | 41 | 38.5 |
| MiniVLA WidowX | moderate | 200 | 64.04 | 61.33 | 2.72 | 4.24 | 56 | 36.0 |
| MiniVLA WidowX | aggressive | 200 | 65.37 | 62.73 | 2.64 | 4.04 | 65 | 33.5 |
| SmolVLA LIBERO Long | strict | 100 | 435.10 | 434.52 | 0.58 | 0.13 | 26 | 40.0 |
| SmolVLA LIBERO Long | moderate | 100 | 434.40 | 432.07 | 2.33 | 0.54 | 64 | 40.0 |
| SmolVLA LIBERO Long | aggressive | 100 | 426.73 | 415.56 | 11.17 | 2.62 | 92 | 42.0 |
| SmolVLA LIBERO Goal | strict | 100 | 151.58 | 151.39 | 0.19 | 0.13 | 10 | 77.0 |
| SmolVLA LIBERO Goal | moderate | 100 | 152.92 | 152.21 | 0.71 | 0.46 | 26 | 76.0 |
| SmolVLA LIBERO Goal | aggressive | 100 | 148.58 | 145.36 | 3.22 | 2.17 | 61 | 79.0 |
| SmolVLA LIBERO Object | strict | 100 | 164.46 | 164.34 | 0.12 | 0.07 | 9 | 89.0 |
| SmolVLA LIBERO Object | moderate | 100 | 165.23 | 164.86 | 0.37 | 0.22 | 18 | 89.0 |
| SmolVLA LIBERO Object | aggressive | 100 | 163.60 | 161.27 | 2.33 | 1.42 | 54 | 90.0 |
| SmolVLA LIBERO Spatial | strict | 100 | 130.84 | 130.81 | 0.03 | 0.02 | 3 | 78.0 |
| SmolVLA LIBERO Spatial | moderate | 100 | 130.85 | 130.52 | 0.33 | 0.25 | 20 | 78.0 |
| SmolVLA LIBERO Spatial | aggressive | 100 | 132.96 | 131.14 | 1.82 | 1.37 | 57 | 76.0 |

Reading notes. (a) The gates open on at most a tenth of the steps (OpenVLA LIBERO, aggressive) and on well under 1 % on CogACT Fractal, SpatialVLA WidowX, CronusVLA WidowX strict and every UniVLA row, so those rows execute the original policy on almost every step; on UniVLA Goal, Object and Spatial no gate ever fired and the three rows are the original run under another name (`results.tex:113-116`). (b) On UniVLA a reuse is counted against a policy call, which on that backbone covers a whole chunk (calls per episode 6.4 for 30 steps). (c) CronusVLA WidowX moderate and aggressive are the 16 September 2026 reruns (`summary.json` field `source` = "new_reruns_widowx (2026-09-16, ...)", `gpus` = one RTX 5090); their summaries carry no `arguments` block, so their thresholds are not recorded in the file, but their gate counts (5 / 33 / 137 reuses over about 9,950 steps) separate the three presets, unlike the superseded runs in which all three had carried the strict thresholds (`DataAudit_2026-09-11.md`, F2). (d) The MiniVLA strict summary carries no `reuse_*` keys (`lens_settings.md`, Section 4); its moderate and aggressive summaries carry the preset values. (e) All SmolVLA reuse rows are the reconstructed implementation (`implementation_versions = ["smolvla-reconstructed-sdpa-v1"]`, no `arguments` block; Section 6). (f) The CronusVLA original, foveation, repeat and depth rows record no per-call latency (only `episode_elapsed_ms`); the reruns do (`timing_samples_ms`).

---

## 5. Temporal fusion

**Selector inputs and their arguments** (`summary.json` `arguments` of the `temporal_fusion_*` folders; values identical on every backbone that records them):

| Symbol in Eq. (8) | Meaning | Record key | motion-entropy | task-aware | conservative-adaptive |
|---|---|---|---|---|---|
| tau_m | motion threshold; patches whose motion exceeds it are protected | `fusion_motion_threshold` | 0.01 | 0.01 | 0.01 |
| eta_e | fraction of patches with the highest normalised entropy that are protected | `fusion_entropy_protect_fraction` | 0.15 | 0.15 | 0.15 |
| eta_r | fraction of patches with the highest text-to-vision attention that are protected | `fusion_task_protect_fraction` | 0.2 | 0.2 | 0.2 |
| Dilate | dilation radius of the protected mask, in patches | `fusion_protect_radius` | 1 | 1 | 1 |
| beta | cap on the reusable fraction of the P patches | `fusion_max_reuse_fraction` | 0.5 | 0.5 | 0.25 |
| keyframe interval | every k-th call recomputes all patches | `fusion_keyframe_interval` | 3 | 3 | 2 |
| forced keyframe | a keyframe is also forced when frame motion exceeds this | `fusion_event_motion_threshold` | none | none | 0.03 |
| relevance r_t collected | text-to-vision attention of the preceding query is read | `fusion_collect_relevance` | false | true | false |
| P_interaction | externally specified interaction region | no record key | not recorded | not recorded | not recorded |

Motion-entropy and task-aware differ only in the relevance term; conservative-adaptive halves the reuse cap, shortens the keyframe interval and adds the forced keyframe (`setup.tex:53-57`). The task-relevance fraction of 0.2 is recorded on every setting, but the relevance term can only act where relevance is collected. No argument in any record corresponds to the interaction region P_interaction of Eq. (8), so whether any external region was supplied is not recorded. Fusion does not change the number of policy calls: `average_policy_calls = average_steps` in every fusion record of a single-action backbone, and `average_reuses = 0` in all of them.

**Where fusion acts** (`summary.json` `fusion.representation`):

| Backbone | Representation fused | Visual tokens per frame implied by the records |
|---|---|---|
| CogACT | "CogACT projected visual tokens before Llama cognition-token generation" | 256 (conservative cap 64 = 0.25 x 256; motion-entropy medians up to 128 = 0.5 x 256) |
| SpatialVLA | "SpatialVLA 16x16 projected visual tokens before Gemma decoding" | 256 |
| UniVLA | "Emu3 VQ-code fusion before UniVLA language-model input", i.e. fusion of the discrete VQ codes before the language model | not stated; medians 300 to 442 (see table) |
| OpenVLA | no `representation` field | 256 (caps 64 and 128 as for CogACT) |
| MiniVLA | no `representation` field | 256 (same caps) |
| SmolVLA | no `fusion` block; per-episode `fusion_reusable_tokens_median` | conservative medians are exactly 16 = 0.25 x 64 |
| CronusVLA | no `fusion` block; per-episode `temporal_fused_patches` | 106 fused patches per call (WidowX) and 112 (Fractal) |

**Task relevance.** The relevance score is the text-to-vision attention of the preceding query (`bagoftricks.tex:232-235`). Collecting it forces an SDPA decoder into eager attention, so the task-aware row of a backbone runs on a different attention backend than its own original row, except on OpenVLA LIBERO, whose original row already runs eager (`setup.tex:108-113`; `main.tex:543-549`). CogACT exposes no text-to-vision attention where fusion runs: `fusion.task_relevance_supported = false` in all 84 CogACT records, no `fusion_collect_relevance` argument, and the CogACT task-aware records are identical to the motion-entropy ones in keyframes, reusable tokens, calls and successes (WidowX 116 of 200 both), so CogACT ran two fusion settings, not three (`DataAudit_2026-09-11.md`, F4). The UniVLA WidowX task-aware row is a 16 September 2026 rerun whose per-episode records count `relevance_queries` equal to `policy_calls` (1,317 each over 200 episodes) and carry no `fusion` block.

**Keyframes and reusable tokens** (`summary.json` `fusion.keyframes`, summed over the task folders of an environment; `fusion.median_reusable_visual_tokens` per task folder, i.e. the median over calls of the number of reused tokens; LIBERO per suite; policy calls pooled from the records):

| Backbone / env | Setting | Calls (total) | Keyframes (total) | Keyframes / calls | Median reusable tokens |
|---|---|---|---|---|---|
| CogACT WidowX | motion-entropy | 9,482 | 3,200 | 0.34 | 108 |
| CogACT WidowX | task-aware (= motion-entropy) | 9,482 | 3,200 | 0.34 | 108 |
| CogACT WidowX | conservative-adaptive | 9,786 | 9,080 | 0.93 | 64 |
| CogACT Fractal | motion-entropy | 11,168 | 3,803 | 0.34 | 85 to 128 (per task) |
| CogACT Fractal | task-aware (= motion-entropy) | 11,168 | 3,803 | 0.34 | 85 to 128 |
| CogACT Fractal | conservative-adaptive | 11,135 | 10,178 | 0.91 | 64 |
| OpenVLA WidowX | motion-entropy | 10,968 | 3,683 | 0.34 | 78 to 128 |
| OpenVLA WidowX | task-aware | 10,106 | 3,404 | 0.34 | 46 to 91 |
| OpenVLA WidowX | conservative-adaptive | 9,704 | 8,977 | 0.93 | 64 |
| OpenVLA Fractal | motion-entropy | 15,508 | 5,256 | 0.34 | 102 to 117 |
| OpenVLA Fractal | task-aware | 15,360 | 5,202 | 0.34 | 47 to 62 |
| OpenVLA Fractal | conservative-adaptive | 15,818 | 13,018 | 0.82 | 64 |
| OpenVLA LIBERO Long | motion-entropy / task-aware / conservative | 41,034 / 40,933 / 40,555 | 13,731 / 13,699 / 34,373 | 0.33 / 0.33 / 0.85 | 128 / 69 / 64 |
| OpenVLA LIBERO Goal | motion-entropy / task-aware / conservative | 15,957 / 16,196 / 15,866 | 5,348 / 5,424 / 14,094 | 0.34 / 0.33 / 0.89 | 128 / 77 / 64 |
| OpenVLA LIBERO Object | motion-entropy / task-aware / conservative | 16,701 / 16,591 / 17,176 | 5,601 / 5,568 / 15,810 | 0.34 / 0.34 / 0.92 | 128 / 65 / 64 |
| OpenVLA LIBERO Spatial | motion-entropy / task-aware / conservative | 12,572 / 13,766 / 13,339 | 4,231 / 4,624 / 12,457 | 0.34 / 0.34 / 0.93 | 124 / 61 / 64 |
| SpatialVLA WidowX | motion-entropy | 9,068 | 3,054 | 0.34 | 57 to 108 |
| SpatialVLA WidowX | task-aware | 9,096 | 3,067 | 0.34 | 26 to 54 |
| SpatialVLA WidowX | conservative-adaptive | 9,138 | 9,000 | 0.98 | 0 |
| SpatialVLA Fractal | motion-entropy | 12,355 | 4,196 | 0.34 | 69 to 118 |
| SpatialVLA Fractal | task-aware | 11,928 | 4,060 | 0.34 | 19 to 48 |
| SpatialVLA Fractal | conservative-adaptive | 12,415 | 12,275 | 0.99 | 0 |
| UniVLA WidowX | motion-entropy | 1,324 | 488 | 0.37 | 314 to 442 |
| UniVLA WidowX | task-aware (rerun) | 1,317 | not recorded | - | not recorded |
| UniVLA WidowX | conservative-adaptive | 1,276 | 1,275 | 1.00 | 0 |
| UniVLA LIBERO Long | motion-entropy / task-aware / conservative | 2,961 / 2,856 / 2,844 | 1,026 / 992 / 2,842 | 0.35 / 0.35 / 1.00 | 312 / 143 / 0 |
| UniVLA LIBERO Goal | motion-entropy / task-aware / conservative | 1,263 / 1,277 / 1,266 | 447 / 452 / 1,266 | 0.35 / 0.35 / 1.00 | 312 / 168 / 0 |
| UniVLA LIBERO Object | motion-entropy / task-aware / conservative | 1,464 / 1,464 / 1,498 | 519 / 522 / 1,498 | 0.35 / 0.36 / 1.00 | 312 / 158 / 0 |
| UniVLA LIBERO Spatial | motion-entropy / task-aware / conservative | 1,172 / 1,170 / 1,127 | 428 / 427 / 1,127 | 0.37 / 0.36 / 1.00 | 300 / 141 / 0 |
| MiniVLA WidowX | motion-entropy | 12,232 | 4,102 | 0.34 | 107 |
| MiniVLA WidowX | task-aware | 12,728 | 4,267 | 0.34 | 68 |
| MiniVLA WidowX | conservative-adaptive | 12,618 | 11,417 | 0.90 | 64 |
| SmolVLA LIBERO (per suite) | motion-entropy | - | not recorded | - | per-episode medians 19 to 23 (Long 23, Goal 21, Object 19, Spatial 22) |
| SmolVLA LIBERO | task-aware | - | not recorded | - | per-episode medians 3.5 to 5 |
| SmolVLA LIBERO | conservative-adaptive | - | not recorded | - | 16 on every episode |
| CronusVLA WidowX / Fractal | all three settings | 10,241 / 12,634 | not recorded | - | 106 / 112 fused patches per call, identical across the three settings |

Reading notes. (a) With interval 3 about one call in three is a keyframe; under conservative-adaptive the forced keyframe at motion 0.03 raises the keyframe share to 0.82 to 0.93 on OpenVLA, CogACT and MiniVLA. (b) On SpatialVLA and UniVLA the conservative-adaptive setting never engages: every call is a keyframe (`fusion.keyframes` equals the call count) and the median reusable count is zero, so those rows are the original policy under the fusion name (`DataAudit_2026-09-11.md`, F3; `results.tex:114-116`). (c) The three CronusVLA fusion settings are one run: their records are identical in success, steps, calls and `temporal_fused_patches` and no fusion argument is recorded (`results.tex:118-120`; `DataAudit_2026-09-11.md`, F2). (d) SmolVLA's fusion rows are the reconstructed implementation; the per-episode `fusion_reusable_tokens_median` values are shown because the summaries have no `fusion` block. (e) The paper's shared mask between fusion and the generation cache (Section III-G) has no counter in the records; no cache-level reuse is measured and fusion is not credited with any acceleration (`bagoftricks.tex:298-303`, commented draft text; `conclusion.tex:37-38` names the measured integration as future work).

---

## 6. Harness details

**Episodes, caps and seeds.**

| Environment | Tasks | Episodes per task | Step cap | Source |
|---|---|---|---|---|
| SimplerEnv WidowX | 4 (spoon on towel, carrot on plate, stack cube, eggplant in basket) | 50 | 60, and 120 for eggplant in basket | `eval.py:74-123` (`max_episode_steps`); record maxima of `steps_executed` per task; `arguments.max_steps = 150` is a harness ceiling above the environment caps |
| SimplerEnv Google Robot / Fractal | 5 (close drawer, move near, open drawer, pick coke can, place apple in closed top drawer) | 50 | 80 | `arguments.max_steps = 80`; record maxima |
| LIBERO Long / Goal / Object / Spatial | 10 each | 10 | 520 / 300 / 280 / 220 | `eval_libero.py:124-130`; `arguments.max_steps`; record maxima equal the caps, so no settle steps are counted in `steps_executed` |

Every configuration of a backbone replays the same task instances and initial states with seed 42 + episode index (`arguments.seed = 42`; `episodes.jsonl` `seed` = 42, 43, 44, ... per task; SpatialVLA and UniVLA WidowX also record `benchmark_episode_id` = episode index and `benchmark_protocol = "released_bridge_visual_matching"`). A failed episode runs to its cap, so the average episode length follows success (`results.tex:100-102`). WidowX runs at control frequency 5 Hz and simulation frequency 500 Hz with the real-inpainting visual-matching overlay, which the in-repo harness refuses to run without (`eval.py:83-84, 362-386`). The in-repo LIBERO harness lets objects settle for 10 no-op steps before the first policy call (`eval_libero.py:131-132, 592-597`), builds one environment per task and resets per trial (`eval_libero.py:553-565`), and latches success at the first step the goal predicate holds (`eval_libero.py:647-653`). Cells: 200 episodes per WidowX configuration, 250 per Fractal configuration, 100 per LIBERO suite and configuration; 22 pairs x 14 configurations = 47,600 episodes (700 summaries).

**Per-backbone run settings recorded in `summary.json`** (`arguments`, `checkpoint_manifest`; checkpoint paths omitted):

| Backbone | Environment(s) | Decoder pruned | Actions per call | Other recorded settings |
|---|---|---|---|---|
| CogACT | WidowX, Fractal | 32 Llama layers | 1 | `cfg_scale 1.5`, `num_ddim_steps 10`; Fractal `policy_setup google_robot`, `action_statistics_key fractal20220817_data` |
| OpenVLA | WidowX, Fractal, LIBERO | 32 Llama-2 layers | 1 | `image_resolution 224`, `center_crop false`; statistics key `bridge_orig` / `fractal20220817_data` / the LIBERO suite (one checkpoint per suite); LIBERO runs `resume: true` |
| SpatialVLA | WidowX, Fractal | 26 Gemma2 layers | chunk 4 (`native_action_chunk_size`), `action_ensemble_temperature -0.8` | `attn_backend sdpa`; statistics key `bridge_orig/1.0.0` / `fractal20220817_data/0.1.0` |
| CronusVLA | WidowX, Fractal | 12-layer DiT action decoder | 1 | records hold `trials 50`, `seed 42`, `depth_indices ""`, the reuse thresholds and, on depth rows, the calibration seed and tasks; no `max_steps`, `device`, per-task block or per-call latency |
| UniVLA | WidowX, LIBERO | 32 Emu3 layers | chunk 5 (WidowX) / 10 (LIBERO) | `attn_backend sdpa`; `environment_action_format libero`, `normalizer_key libero` on LIBERO; LIBERO runs `resume: true` |
| MiniVLA | WidowX | 24 Qwen2.5 layers | 1 | `image_resolution 224`, `center_crop false`, statistics key `bridge_dataset`, base VLM `prism-qwen25-extra-dinosiglip-224px+0_5b` |
| SmolVLA | LIBERO | 32 SmolLM2 layers (fixed indices, Section 3) | 1 | legacy rows: `transformers 4.51.3 (isolated)`, `lerobot 0.4.4`, VLM SmolVLM2-500M-Instruct; reconstructed rows: `implementation_versions ["smolvla-reconstructed-sdpa-v1"]`, no `arguments` |

Decoder depths and the chunk sizes are as stated in `setup.tex:21-30`.

**Attention backends and library versions** (`experiments/paper/AttentionBackend_mentor.md`, the authors' note of 2026-09-08; records where they exist):

| Backbone | Attention backend | Library versions | Source |
|---|---|---|---|
| CogACT | Llama decoder resolves implicitly to SDPA under transformers 4.47.0; the DiT action head has its own attention | transformers 4.47.0 | note lines 9, 29-41 |
| CronusVLA | Qwen2.5 decoder resolves implicitly to SDPA under transformers 4.47.0; the action decoder uses timm's fused attention (PyTorch SDPA) when available | transformers 4.47.0 | note lines 10, 43-57 |
| MiniVLA | implicit, version-dependent; observed as SDPA under 4.47.0 although MiniVLA's own environment pins 4.40.1 | transformers 4.47.0 | note lines 11, 59-71 |
| SpatialVLA | mixed: the custom Gemma2 text decoder falls back to eager because of logit soft-capping; SigLIP and ZoeDepth are separate components | transformers 4.47.0 (`main.tex:471-473`) | note lines 8, 14-27; records `attention_backend = "native_mixed (SDPA where supported; ZoeDepth compatible)"`, `attn_backend = "sdpa"` |
| SmolVLA | explicit call to PyTorch `scaled_dot_product_attention`; the kernel is chosen by PyTorch, FlashAttention-2 not guaranteed | transformers 4.51.3 (isolated), lerobot 0.4.4 (legacy rows) | note lines 12, 73-87; `checkpoint_manifest` |
| OpenVLA | SDPA requested on SimplerEnv; the LIBERO wrapper defaults to eager | not recorded | note line 1; `inference_openvla_libero.py:122` (`attn_implementation="eager"`); `setup.tex:29-30` |
| UniVLA | SDPA requested (`attn_implementation="sdpa"`) | not recorded | `inference.py:796`; `inference_libero.py:204`; records `attention_backend = "sdpa"` |

No record and no note gives a torch version (`lens_settings.md:27`). The in-repo OpenVLA LIBERO wrapper targets the transformers 4.40.x API (`inference_openvla_libero.py:57-65`); the transformers versions of the OpenVLA and UniVLA runs were left for the authors to fill in the draft (`main.tex:473`).

**Hardware, as far as the records say.** The GPU is recorded only in the episode records that carry a `gpu` field:

| Runs | GPU(s) recorded | Source |
|---|---|---|
| CronusVLA WidowX guarded reuse moderate and aggressive (16 September 2026 reruns) | NVIDIA GeForce RTX 5090 (all 200 episodes each) | `episodes.jsonl` `gpu`; `summary.json` `gpus`, `source` |
| UniVLA WidowX temporal fusion task-aware (16 September 2026 rerun) | NVIDIA GeForce RTX 5090 (all 200) | same |
| SmolVLA guarded reuse x3 and temporal fusion x3, all suites | NVIDIA GeForce RTX 5090 (all 400 per configuration) | `episodes.jsonl` `gpu`, `implementation` |
| SmolVLA depth pruning, 2 layers, Long and Goal (reconstructed episodes only, 43 of 200) | RTX 5090 (33), RTX PRO 6000 Blackwell (10) | same |
| SmolVLA depth pruning, 4 layers, Long, Goal, Object (reconstructed episodes, 262 of 300) | RTX 5090 (186), RTX 6000 Ada (46), L40S (18), RTX A6000 (12) | same |
| every other run | no GPU field | - |

The SmolVLA reconstructed evaluator's per-call time is about 5 percent lower than the legacy SmolVLA implementation on the same card (`experiments/paper/project_page/pvalues.md:5`), and its legacy and reconstructed rows differ in implementation and hardware, so their latency changes are not matched comparisons (`summary.json` `claim_scope` of the reconstructed rows; `setup.tex:98-103`). See Discrepancies, item D6, for the paper's hardware sentence.

**Latency definition.** The paper's latency is wall-clock per environment step, computed as the mean episode time divided by the mean episode length, `1000 x avg_episode_time_s / avg_steps` (`experiments/make_simpler_table.py:17, 42-43`; `setup.tex:62-67`). The episode clock (`episodes.jsonl` `episode_elapsed_ms`, `summary.json` `average_episode_ms`) runs from the episode's reset to its termination or cap, so it **includes** simulator stepping and rendering, the policy calls, the foveation blur, the reuse gate's signature computation, the fusion selector, and, for the six backbones calibrated on the first frame, the unpruned calibration call. It **excludes** model loading, environment construction, CronusVLA's separate calibration pass, and any video or record writing outside the loop. Speedup in the figures is the original's time per step divided by the trick's (paper, Fig. 5 caption). Every harness other than CronusVLA also records the per-call model time (`query_latency_ms` per episode, `policy_query_latency_ms` in the summary); the CronusVLA original, foveation, repeat and depth rows have only the episode clock, and the two CronusVLA reruns record `timing_samples_ms`. The `inference_cycle_ms` field has a harness-specific meaning (per environment step on some harnesses, per call spanning a chunk on UniVLA, equal to the per-call time on CronusVLA) and is not used (`DataAudit_2026-09-11.md`, R1). Latency is read only as the change from the original within one backbone, one environment, one software stack and one GPU class (`setup.tex:65-67`).

**Reproducibility floor.** Episodes on which no reuse gate fired run the original policy at every step; their outcome matches the original run on all but at most one episode in a hundred on SpatialVLA, UniVLA, OpenVLA Fractal and OpenVLA LIBERO, and differs on 2 to 17 percent of such episodes on CogACT (WidowX 16.5 %, Fractal 1.8 %), CronusVLA (WidowX 11.3 %, Fractal 4.6 %), MiniVLA (9.1 %) and OpenVLA WidowX (13.4 %), and on 19.9 % for the reconstructed SmolVLA rows (`setup.tex:84-91` and its verification comment). On those pairs a change of a few points is within run-to-run noise.

---

## 7. Discrepancies and caveats

D1. **Depth-pruning protection rule: code default versus records.** The in-repo `DepthPruner` protects the first half of the stack (`min_layer = 0.5`, `depth_prune.py:126, 231-234`; `eval_libero.py:273-275`) and does not protect the final layer (candidates run to `n - 1`, `depth_prune.py:234`). The records, and the paper's Section III-D, follow a quarter-of-the-stack window with the final layer protected and no adjacent removals, stated explicitly only in the CronusVLA records (`rules = {min_layer_fraction 0.25, protect_last 1, min_gap 1}`) and satisfied by every calibrated selection (Section 3). Selections such as OpenVLA Spatial [8], SpatialVLA [8, 10] and MiniVLA [7, 9, 11, 13] are impossible under the in-repo default. The harness that wrote the records is not in this repository. Also, Eq. (4) averages over N calibration examples; for all backbones except CronusVLA the record holds one influence array measured on the first observation of the evaluation run, which is itself one of the test episodes, whereas an earlier draft described the calibration frames as disjoint from the test episodes (`main.tex:493-495`). The paper's phrasing "calibration set" is consistent with the records; CronusVLA's pass uses a separate seed (10000) over the same task list.

D2. **Foveation variant and views.** The in-repo CLI default is the log-polar variant (`eval.py:195`, `eval_libero.py:227`); the paper's Eq. (2) is the blur variant, which must be selected explicitly. The records name only `fovea_keep_ratio`, not the variant, the centre or, on LIBERO, the views. The in-repo LIBERO harness leaves the wrist view unfoveated by default (`eval_libero.py:246-253`); whether the reported UniVLA and SmolVLA LIBERO foveation runs foveated the wrist view is not recorded.

D3. **tau at the boundary.** The paper says tau(x) reaches one at the image boundary; in the code it reaches one at the farthest corner (`foveation.py:118-126`), so the midpoints of the image edges are blended from G3 and G9 rather than pure G9.

D4. **CogACT task-aware equals motion-entropy.** CogACT exposes no text-to-vision attention where fusion runs (`fusion.task_relevance_supported = false`), so its task-aware row is a duplicate of motion-entropy and CogACT tested two fusion settings, not three (Section 5). The submitted paper's Section IV-A counts thirteen settings for every backbone without this qualification; the draft carried it (`setup.tex:108-110`).

D5. **Settings that never engaged or were not recorded.** Conservative-adaptive fusion never reuses a patch on SpatialVLA and UniVLA (keyframe on every call), so the Table I UniVLA fusion cell and the Table II UniVLA Object and Spatial fusion cells report the original policy under the fusion name; the UniVLA guarded-reuse rows on Goal, Object and Spatial and the strict row on WidowX fired no gate and equal the original run. The three CronusVLA fusion settings are one run with unrecorded settings, so the CronusVLA fusion cells in Table I are that single run. The 16 September 2026 reruns (CronusVLA WidowX moderate and aggressive reuse, UniVLA WidowX task-aware fusion, all SmolVLA reuse and fusion rows) have no `arguments` block, so their thresholds are not in the files.

D6. **Hardware.** The paper's Section IV-A says the runs used a shared GPU cluster and "a system equipped with 4x RTX 5090 GPUs". The records name a GPU only for the reruns and the reconstructed SmolVLA rows (Section 6): the reruns are on an RTX 5090, while the reconstructed SmolVLA depth rows span RTX 5090, RTX PRO 6000 Blackwell, RTX 6000 Ada, L40S and RTX A6000, mixed inside single folders. The draft setup text placed the three reruns on an RTX PRO 6000 (`setup.tex:95-98`; `DataAudit_2026-09-11.md`, update of 2026-09-12), whereas the record files of those reruns say RTX 5090 (`summary.json` `gpus`, `source`; `pvalues.md:5`). No other run records its card.

D7. **SmolVLA depth pruning is not the Section III-D procedure.** Its removed layers are fixed indices with no influence measurement (`calibrated: false`), part of its depth rows were completed under a second implementation on other cards, and its legacy per-call latency does not fall with the number of removed layers (`DataAudit_2026-09-11.md`, F5). The draft table printed the SmolVLA reuse and fusion latencies as "--" (`tablelibero.tex:7-9`) while the submitted Table II prints values for them.

D8. **Reuse bookkeeping under action repeat differs by harness** (Section 2): `reuses` is 0, `steps - calls`, or `(k - 1) x chunk x calls` depending on the backbone, and on UniVLA LIBERO it exceeds the episode length. Only `policy_calls` and `steps_executed` are comparable.

D9. **Bypass versus structural removal.** A draft of Section III-D said the network is "structurally shortened rather than emulated with identity layers" (`bagoftricks.tex:131-133`); the implementation swaps in a pass-through module that skips the block's computation and writes a zero KV placeholder so the cache stays contiguous (`bypass_layer.py:17-62`). The submitted text ("reduces decoder computation while maintaining generation-cache consistency") matches the implementation.

D10. **VLA-Cache and the shared mask.** Section III-F describes VLA-Cache and Section III-G a mask shared with the generation cache, but no configuration among the fourteen runs VLA-Cache, and no record counts cache-level reuse; fusion reduces neither policy calls nor, by the records, per-call time.

D11. **Interaction region.** Eq. (8) includes an externally specified interaction set P_interaction; no argument or counter in the records corresponds to it.

D12. **Attention backend of the task-aware setting.** Collecting relevance forces eager attention, so on MiniVLA and UniVLA the task-aware per-call time mixes the relevance cost with a backend change; only on OpenVLA are the two separated, its LIBERO original row already running eager (`setup.tex:110-113`). This qualification is in the draft but not in the submitted paper.

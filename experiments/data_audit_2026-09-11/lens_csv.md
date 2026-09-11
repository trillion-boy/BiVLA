All six lenses are computed. No further tool calls are needed; here is the report.

# Audit of `artifacts/results/mentor_2026-09-11/summarization/` (25 CSVs opened: simpler_widowx 6+1, google_robot_fractal 4+1, libero 12+1)

Script: `/tmp/claude-0/-home-user-BiVLA/4327cefa-ba5d-5e2e-9d87-c73dd3a8ccbb/scratchpad/audit.py` (full output in `audit_out.txt` alongside). Read-only; nothing under the repo was touched.

## 1. Structure

| file | data rows | cols | line endings | N/A cells | empty cells |
|---|---|---|---|---|---|
| simpler_widowx/{cogact_base,cronusvla,minivla,openvla_simplerenv,spatialvla,univla}_*.csv | 14 each | 23 | CRLF (all 15 lines) | 33 each | 0 |
| simpler_widowx/summary.csv | 36 | 23 | CRLF | 90 | 0 |
| google_robot_fractal/{cogact_base,cronusvla,openvla,spatialvla}_simplerenv_fractal.csv | 14 each | 23 | CRLF | 33 each | 0 |
| google_robot_fractal/summary.csv | 24 | 23 | CRLF | 60 | 0 |
| libero/{openvla,smolvla,univla}_libero_{goal,object,spatial}.csv, smolvla/univla_libero_10.csv | 14 each | **24** | CRLF | 33 each | 0 |
| **libero/openvla_libero_10.csv** | **13** | 24 | CRLF | 32 | 0 |
| libero/summary.csv | 72 | 23 | CRLF | 180 | 0 |

- Every file is CRLF-terminated with a trailing CRLF, no BOM, no LF-only lines, no empty cells.
- Column sets: the 12 LIBERO per-suite files add `suite` at position 3 (after `checkpoint_model_id`); everything else is the 23-column base header in identical order. `libero/summary.csv` **drops `suite`** and encodes it in `environment` instead (see section 4).
- N/A pattern is deterministic: 3 N/A per non-depth/non-fusion row (`selected_depth_layers`, `fusion_keyframe_interval`, `fusion_max_reuse_fraction`), 2 per depth row, 1 per fusion row → 8×3+3×2+3×1 = 33 per 14-row file. `median_reusable_visual_tokens` is never N/A (it is `0`/`0.0` in non-fusion rows).
- Numeric-as-text inconsistencies (all columns parse as numbers, but textual form is mixed across files, so `==`-style joins or string sorts will break):
  - `success_rate_pct`: 300 int-form (`50`), 139 float-form (`36.0`). minivla always float; LIBERO always int except `0.0` for zero-success rows; widowx/fractal mixed per row.
  - `avg_reuses`: `0` (cronusvla, openvla_widowx, spatialvla, univla_widowx, all fractal) vs `0.0` (cogact_widowx, minivla, all LIBERO).
  - `median_reusable_visual_tokens`: `0` (cronusvla, smolvla) vs `0.0` elsewhere; fusion rows mix `108`, `107.4`, `64`, `64.0`, `403.625`.
  - Integer-form floats scattered: `avg_steps` `75`/`520`/`300`/`220` (4 rows), `avg_episode_time_s` `6`/`72`/`124`/`57`/`28`/`23` (9 rows), `policy_median_latency_ms` `171`, `cycle_median_latency_ms` `168`, `cycle_p95_latency_ms` `321`, `control_frequency_hz` `6`.
  - Precision differs: cronusvla writes 3 decimals for `avg_episode_time_s`/`control_frequency_hz` (`6.362`, `6.828`), other SIMPLER backbones 2; LIBERO hz has 3.
- `selected_depth_layers` separator: `;` in 20 files; **`,` (and therefore double-quoted) in the two cronusvla files** (`"8,10"`, `"4,6,8,10"`, `"3,6,8,10"`). Both summaries only happen to include cronusvla depth_pruning1 (single value `10`), so the quoted form never reaches a summary.
- `minivla_simpler_widowx.csv` rows are in alphabetical configuration order (original, action_repeat2, action_repeat4, depth_pruning…, fixed_foveation…, guarded_reuse_aggressive/moderate/strict, temporal_fusion_conservative_adaptive/motion_entropy/task_aware); all other 21 per-backbone files use the canonical family order. Summaries re-sort into canonical order.

## 2. Column semantics by harness (evidence from the numbers)

### 2a. `cycle_median_latency_ms` vs `policy_median_latency_ms` (non-action-repeat rows)

| backbone (files) | cycle − policy, ms | cycle/policy | reading |
|---|---|---|---|
| CronusVLA widowx | 0.000 (12/12 exactly equal) | 1.000 | cycle **is** policy latency (no env time) |
| CronusVLA fractal | −3.55..0.00 (9/12 equal; 3 guarded rows cycle < policy) | 0.977..1.000 | same; guarded rows mix reuse steps into the median |
| CogACT widowx / fractal | +39.6..+62.0 / +63.7..+87.8 | 1.40..1.63 / 1.64..1.88 | cycle = policy + env step (~40–90 ms) |
| SpatialVLA widowx / fractal | +38.9..+63.9 / +62.6..+97.1 | 1.10..1.17 / 1.17..1.26 | same |
| OpenVLA widowx / fractal | +7.3..+14.8 / +7.7..+18.4 | 1.05..1.09 / 1.05..1.11 | cycle = policy + ~10 ms |
| OpenVLA LIBERO (4 suites) | +2.3..+8.2 | 1.014..1.052 | cycle ≈ policy + 2–8 ms |
| MiniVLA widowx | +7.7..+14.6 | 1.08..1.15 | |
| SmolVLA LIBERO | +12.6..+34.3 | 1.05..1.12 | |
| UniVLA widowx / LIBERO | +173..+213 / +108..+195 | 1.20..1.30 / 1.22..1.40 | cycle spans one **call = one chunk** (4.7 / 9.7 env steps) |

### 2b. What happens under `action_repeat` (three incompatible semantics)

| harness | cycle under AR2 / AR4 vs original | policy under AR | avg_reuses | meaning of cycle |
|---|---|---|---|---|
| CogACT widowx | 179.7 / 257.9 vs 139.6 (**rises**, ×1.29/×1.85) | flat (98.6) | **0** (steps−calls = 35.3 / 55.7 not booked) | per call, includes all repeated env steps |
| CogACT fractal | 230.8 / 398.8 vs 162.7 (×1.42/×2.45) | flat | 0 | same |
| SpatialVLA widowx / fractal | 464 / 539 vs 421; 520 / 656 vs 447 (rises) | flat | 0 | same |
| UniVLA widowx / LIBERO | 1116 / 1497 vs 902; e.g. libero_10 700 / 951 vs 574 (rises) | ~flat | = calls×chunk×(repeat−1) exactly (7.25×5×1 = 36.25; 3.74×5×3 = 56.1; libero_goal AR4 8×10×3 = 240) | per call incl. chunk × repeat |
| CronusVLA widowx / fractal | 81.45 / 59.93 vs 126.3 (= policy/2, policy/4 exactly, ratios 0.500/0.250); fractal 0.500/0.251 | **rises** 162.8 / 239.7 vs 126.3 (per call incl. repeats) | = steps − calls exactly | amortised per env step |
| OpenVLA widowx / fractal | 86.8 / 47.5 vs 167.1 (0.551× / 0.300×); fractal 0.550× / 0.299× | flat | = steps − calls exactly | amortised per env step |
| OpenVLA LIBERO | AR2 ≈ policy (157.2 vs 156.7); **AR4 = 2.24 ms** (all 4 suites 2.240–2.249) | flat | = steps − calls | per-step median: with 75 % reuse steps the median is a reuse step |
| MiniVLA widowx | AR2 103.6 (≈ original 104.9); **AR4 7.74** | flat | = steps − calls | same per-step median artefact |
| SmolVLA LIBERO | AR2 283–291 (≈ original 289–294); **AR4 12.5–18.9** | flat | = steps − calls | same |

Consequences: `cycle_median_latency_ms` is not comparable across backbones in AR rows (rises ×2.45 for CogACT, falls to 1.4 % of policy for OpenVLA-LIBERO). Rows where cycle < policy: 26 (all AR rows of cronusvla/openvla-SIMPLER/OpenVLA-LIBERO-AR4/MiniVLA-AR4/SmolVLA-AR4, plus the 3 cronusvla-fractal guarded rows, plus the copies in the two SIMPLER summaries). No p95 < median anywhere.

Under `action_repeat`, UniVLA `avg_reuses` counts chunk-expanded repeats without horizon truncation: libero_goal AR4 has avg_steps = 300 (the horizon) but calls×chunk×(repeat−1)+calls×chunk = 320 > 300 (reuses = 240 vs steps − calls = 292).

### 2c. `control_frequency_hz` definition (max relative error over all rows of the file)

| backbone | 1000/cycle_med | steps/t | calls/t | verdict |
|---|---|---|---|---|
| OpenVLA widowx / fractal | 4.2 % / 2.3 % | 47 % / 59 % | 87 % / 90 % | ≈ **1000/cycle** (env time outside the cycle is ~43 ms/step, so steps/t ≠ hz) |
| OpenVLA LIBERO ×4 | 23–26× off (AR4) | **0.02–0.04 %** | 75 % | **steps / episode_time** exactly |
| MiniVLA | 8.6× off | **0.09 %** | 75 % | steps/t exactly |
| SmolVLA ×4 | 3.6–5.5× off | 1.2–2.3 % | 75 % | ≈ steps/t (worst row smolvla_libero_object temporal_fusion_motion_entropy: hz 3.116 vs 3.186) |
| CronusVLA widowx / fractal | 3.8 % / 0.9 % | 2.6 % / 1.7 % | 76 % / 75 % | ≈ 1000/cycle ≈ steps/t, neither exact (probably mean-based) |
| CogACT widowx / fractal | 4.8 % / 4.6 % | 3.0× (AR rows) | **2.7 % / 2.6 %** | **calls / episode_time** (= steps/t only when no repeat). AR2: hz 5.44 = 35.33/6.5 = 5.435; AR4: 3.7 = 3.700 |
| SpatialVLA widowx / fractal | 0.8 % / 0.6 % | 2.9× (AR) | 0.6 % / 2.5 % | calls / episode_time (AR2 2.14 = 24.85/11.61 = 2.140) |
| UniVLA widowx / LIBERO | 2.7 % / 2–4 % | 18.6× / 36–39× | **4.5 % / 0.06–2.1 %** | **calls / episode_time** = decode-call rate; libero_10 1.695 = 28.44/16.78 exactly |

So `control_frequency_hz` means "env steps per second" for OpenVLA-LIBERO/MiniVLA/SmolVLA/CronusVLA, "1/cycle" for OpenVLA-SIMPLER, and "policy calls per second" for CogACT/SpatialVLA/UniVLA. Under action_repeat the same column therefore **rises** for OpenVLA/Cronus/MiniVLA/SmolVLA (5.97→11.45→20.8) and **falls** for CogACT/SpatialVLA/UniVLA (7.07→5.44→3.7; 2.36→2.14→1.84; 1.1→0.89→0.65). For UniVLA the real env-step rate is 30.21/5.77 = 5.24 Hz (widowx) and 16.7 Hz (libero_10), not the reported 1.1 / 1.695.

### 2d. `avg_reuses` accounting

| family | CogACT / SpatialVLA | CronusVLA / OpenVLA (both) / MiniVLA / SmolVLA | UniVLA |
|---|---|---|---|
| action_repeat | **0** although steps ≠ calls | = steps − calls (|Δ| ≤ 0.011) | = calls×chunk×(repeat−1) (see 2b) |
| guarded_reuse | = steps − calls | = steps − calls | ≤ 0.17; 0 in libero goal/object/spatial and widowx strict (and there steps, calls, success are byte-identical to `original`) |
| original / foveation / depth / fusion | 0 and steps == calls | same | 0 but steps ≠ calls (chunking is not counted as reuse) |

### 2e. `action_repeat` / `depth_layers` defaults in unrelated rows

Every non-action-repeat row carries `action_repeat=2` and every non-depth row `depth_layers=1` — in all 20 non-cronusvla files (11–12 rows each). Only the two cronusvla files use the semantically "off" values `action_repeat=1`, `depth_layers=0`. Hence the `original` row of OpenVLA/CogACT/… literally says `action_repeat=2`, indistinguishable from the `action_repeat2` row by these columns alone; the configuration name is the only reliable key.

### 2f. Fusion columns

- Non-fusion rows: `fusion_keyframe_interval=N/A`, `fusion_max_reuse_fraction=N/A`, `median_reusable_visual_tokens=0`/`0.0` (should be N/A for consistency with the other two).
- Fusion rows: motion_entropy / task_aware = (3, 0.5), conservative_adaptive = (2, 0.25) everywhere **except CronusVLA, where all three variants are (1, 0.5)** with identical tokens (110.542 widowx, 115.285 fractal) and identical successes/steps (33.5/51.205 ×3; 57.2/50.536 ×3) — the three "variants" are the same run configuration for CronusVLA. CogACT motion_entropy and task_aware are also identical in success/steps/tokens (58/47.41/108 widowx; 64.8/44.672/107.4 fractal). SmolVLA reports `median_reusable_visual_tokens=0` in all 12 fusion rows. conservative_adaptive has tokens 0.0 for SpatialVLA and UniVLA (both envs) and is byte-identical to `original` in success/steps for univla widowx (87.5/30.21), univla libero_object (96/145.82) and libero_spatial (95/108.31) — effectively a no-op there.

## 3. summary.csv vs per-backbone files

| summary | rows | byte-equal to a per-file row | field-equal (all cols except environment) | rows not found | families complete (6 per backbone×env) |
|---|---|---|---|---|---|
| simpler_widowx/summary.csv | 36 | **36/36** | 36/36 | 0 | yes (6×6) |
| google_robot_fractal/summary.csv | 24 | **24/24** | 24/24 | 0 | yes (4×6) |
| libero/summary.csv | 72 | **0/72** (suite column dropped, environment renamed) | **72/72** | 0 | yes (3 backbones × 4 suites × 6) |

Selection rule check ("highest success_rate_pct, then fewer avg_steps"):

- **google_robot_fractal**: 0 deviations. Ties broken by file order where success and steps tie (cronusvla temporal_fusion: 3 identical rows → motion_entropy = first in file; min-cycle would give conservative_adaptive 154.181, min-policy also conservative_adaptive).
- **simpler_widowx**: 1 deviation. `cronusvla_simplerenv_bridge` depth_pruning: summary has **depth_pruning1 (35.5 %, 50.025 steps, 122.1 ms)** although depth_pruning2 has 36 % (50.52 steps, 119.1 ms). No tested rule reproduces it (max-success→min-steps, →min-policy-latency, →min-cycle-latency, →min-episode-time, →file-order all pick depth_pruning2). Looks like a manual or stale pick. All other 30 picks match; cronusvla guarded_reuse (3-way tie on success and steps) resolves to strict = first in file (min-cycle would give moderate 123.876).
- **libero**: 8 deviations, and in every one the summary picked the tied-on-success variant with the **lower `cycle_median_latency_ms`** rather than fewer steps. Rule "max success → min cycle_median_latency_ms" fits all 72 LIBERO picks; "→ min policy_median" fails 1, "→ min avg_episode_time_s" fails 2, "→ file order" fails 6.

| model | env (summary) | family | picked (succ / steps / cycle) | min-steps alternative (succ / steps / cycle) |
|---|---|---|---|---|
| openvla_libero | Libero Long | guarded_reuse | strict 51 / 397.87 / 159.795 | moderate 51 / 396.62 / 159.916 |
| openvla_libero | LIBERO-Object | guarded_reuse | aggressive 85 / 167.1 / 160.175 | strict 85 / 165.41 / 161.312 |
| smolvla_libero | Libero Long | temporal_fusion | motion_entropy 44 / 415.45 / 274.693 | conservative_adaptive 44 / 414.93 / 275.218 |
| smolvla_libero | LIBERO-Spatial | guarded_reuse | moderate 79 / 129.59 / 281.889 | strict 79 / 129.2 / 282.809 |
| univla_libero | Libero Long | guarded_reuse | aggressive 86 / 281.23 / 558.819 | strict 86 / 280.58 / 570.434 |
| univla_libero | LIBERO-Goal | guarded_reuse | moderate 93 / 121.54 / 612.278 | strict 93 / 121.34 / 619.65 |
| univla_libero | LIBERO-Object | guarded_reuse | aggressive 96 / 145.82 / 587.873 | strict 96 / 145.82 / 591.479 (steps tie; file order) |
| univla_libero | LIBERO-Spatial | guarded_reuse | aggressive 95 / 108.31 / 686.978 | strict 95 / 108.31 / 692.363 (steps tie; file order) |

So the two SIMPLER summaries and the LIBERO summary were built with different tie-breaks (steps+file order vs cycle latency), plus one unexplained pick in widowx.

Picks per family (for the Results text): widowx — fixed_foveation keep20 for cogact/cronus/spatialvla/univla, keep50 for minivla/openvla; action_repeat2 everywhere (all envs); depth_pruning2 for cogact/openvla/univla, 1 for cronus/minivla/spatialvla; guarded strict (cronus/minivla/openvla), moderate (univla), aggressive (cogact/spatialvla); temporal motion_entropy ×4, task_aware (openvla), conservative_adaptive (univla). fractal — keep50 ×4; depth 4/1/2/2 (cogact/cronus/openvla/spatialvla); guarded moderate/aggressive/moderate/moderate; temporal conservative_adaptive/motion_entropy/task_aware/task_aware. libero — keep50 in 11/12 (univla Object: keep20, 100 %); depth_pruning1 in 11/12 (univla Long: 2); see table for guarded/fusion.

## 4. Naming

- `model_name` (13): `openvla_simplerenv`, `minivla_simplerenv` (no env suffix) vs `cogact_base_simplerenv_bridge`, `cronusvla_simplerenv_bridge`, `spatialvla_4b_simplerenv_bridge`, `univla_simplerenv_bridge`; fractal: `{cogact_base,cronusvla,openvla,spatialvla_4b}_simplerenv_fractal`; LIBERO: `openvla_libero`, `smolvla_libero`, `univla_libero`. File names disagree with model names (`openvla_simplerenv_widowx.csv` vs the other `*_simpler_widowx.csv`; `spatialvla_simpler_widowx.csv` for `spatialvla_4b_…`).
- `environment` (7): `simpler_widowx`; `simpler_google_robot_fractal` (directory is `google_robot_fractal`); `libero` in all 12 per-suite files; and in `libero/summary.csv` only: **`Libero Long`, `LIBERO-Goal`, `LIBERO-Object`, `LIBERO-Spatial`** (mixed case/style; `libero_10` → "Libero Long" with a space, others hyphenated upper-case). Joining summary back to per-suite files needs an explicit map.
- `suite` (per-suite files only): `libero_10`, `libero_goal`, `libero_object`, `libero_spatial`.
- `checkpoint_model_id` (12): `openvla-7b` used for **both** widowx and all LIBERO files (different fine-tunes, same id); `openvla-7b-Fractal`; `CogACT-Base-Bridge` / `CogACT-Base-Fractal`; `CronusVLA-0.5B-Bridge-RT-1` / `CronusVLA-0.5B-Bridge-RT-1-Fractal`; `spatialvla-4b-bridge` vs `spatialvla-4b-224-sft-fractal` (different conventions); `prism-qwen25-extra-dinosiglip-224px+0_5b` (MiniVLA internal name); `SmolVLA-LIBERO` (not a hub id); UniVLA widowx = hub-style `Yuqi1997/UniVLA/UNIVLA_SIMPLER_BRIDGE_VIDEO_BS128_20K`, UniVLA LIBERO = **absolute local path** `/home/soumyara004/soumyaratnadebnath/VisionLanguageAction/models/UniVLA/weights/UNIVLA_LIBERO_VIDEO_BS192_8K` (leaks a user home directory; appears in 5 files incl. summary).
- Summary row order: widowx univla, cogact, openvla, spatialvla, cronusvla, minivla; fractal cogact, openvla, spatialvla, cronusvla; libero univla, openvla, smolvla — none alphabetical.

## 5. latency_per_step = 1000·avg_episode_time_s / avg_steps

Original values and per-family change relative to original (min..max over the family's variants; LIBERO = range over 4 suites):

| backbone / env | original ms/step | fixed_foveation | action_repeat (AR2 / AR4) | depth_pruning | guarded_reuse | temporal_fusion |
|---|---|---|---|---|---|---|
| CogACT widowx | 141.4 | +13.0..+15.1 % | −34.9 / −52.2 % | −4.2..−1.9 % | −2.0..−0.1 % | +0.0..+0.5 % |
| CogACT fractal | 169.1 | +12.2..+15.1 % | −28.1 / −42.3 % | −3.0..+0.7 % | −0.2..+4.2 % | +1.2..+4.3 % |
| CronusVLA widowx | 125.8 | +15.9..+16.8 % | −32.7 / −49.3 % | −9.9..−2.9 % | −1.5..−0.3 % | +0.9..+1.5 % |
| CronusVLA fractal | 155.1 | +14.9..+16.0 % | −27.6 / −42.1 % | −7.1..−1.5 % | −3.1..−1.3 % | +0.8..+1.1 % |
| OpenVLA widowx | 210.2 | +2.8..+3.6 % | −38.0 / −57.0 % | −9.6..−2.7 % | −5.0..−3.6 % | −1.0..+5.7 % |
| OpenVLA fractal | 239.8 | +3.8..+4.0 % | −33.6 / −50.4 % | −6.9..−1.4 % | −2.1..−1.3 % | +0.2..+5.9 % |
| OpenVLA LIBERO (10/goal/object/spatial) | 171.6 / 176.1 / 173.9 / 181.0 | +2.0..+4.1 % | −44.1..−46.1 / −66.1..−68.7 % | −11.2..−1.8 % | −10.4..−2.0 % | −0.9..+8.2 % |
| SpatialVLA widowx | 423.5 | +5.2..+5.3 % | −44.5 / −67.5 % | −11.6..−3.6 % | −0.2..+0.4 % | +0.7..+1.1 % |
| SpatialVLA fractal | 450.0 | +5.5..+6.3 % | −41.6 / −63.1 % | −11.3..−3.3 % | −2.0..−0.7 % | +0.3..+0.7 % |
| MiniVLA widowx | 149.4 | +4.7..+7.2 % | −33.6 / −50.4 % | −12.2..−2.4 % | −3.1..−2.1 % | −0.5..+8.4 % |
| SmolVLA LIBERO (10/goal/object/spatial) | 289.5 / 295.7 / 290.9 / 292.3 | +2.8..+7.4 % | −46.0..−48.4 / −70.1..−72.2 % | −16.7..+2.2 % | −13.6..+3.6 % | −7.9..+7.9 % |
| UniVLA widowx | 191.0 (= 902 ms/call ÷ 4.74 steps/call) | −0.8..+0.4 % | −40.6 / −58.9 % | −9.4..−3.0 % | −0.7..0.0 % | −1.1..+16.9 % (task_aware +16.9 %) |
| UniVLA LIBERO (10/goal/object/spatial) | 59.9 / 65.1 / 62.4 / 72.3 | −0.0..+3.1 % | −35.8..−40.0 / −54.2..−60.2 % | −9.1..−0.7 % | −2.7..+1.4 % | −2.2..+12.0 % |

Notes: foveation is consistently *slower* per step (+2 % to +17 %); depth pruning is the only non-repeat family that reduces per-step time on every backbone (up to −16.7 %, SmolVLA object dp4, at 13 % success). Episode-time overhead outside `cycle`: ≈ 0–2 ms/step for CogACT/Cronus/UniVLA (t/calls ≈ cycle), ≈ 43 ms/step for OpenVLA-SIMPLER (210 vs 167), ≈ 13 ms for OpenVLA-LIBERO.

Rows with `avg_policy_calls ≠ avg_steps` (155 of 308 per-backbone rows):
- All 44 action_repeat rows: calls/steps = 0.500–0.504 (AR2), 0.250–0.253 (AR4) for every backbone except UniVLA: widowx 0.101 / 0.050, LIBERO 0.050–0.051 / 0.025–0.027 (chunk × repeat).
- All 66 guarded_reuse rows: calls/steps 0.888 (OpenVLA libero_goal aggressive) … 0.9998 (SpatialVLA widowx strict); UniVLA LIBERO goal/object/spatial guarded rows and univla widowx strict have reuses = 0 but are still ≠ because of chunking.
- All 70 UniVLA rows (only chunked-execution backbone): widowx calls/steps 0.207–0.211 (4.74–4.82 steps per call), LIBERO 0.101–0.104 (9.6–9.9 steps per call). CogACT and SmolVLA (natively chunk-predicting models) report calls == steps in all non-repeat rows, i.e. the harness queries them every step.

## 6. Sanity

- `success_rate_pct == successes/episodes·100`: holds in all 439 rows (0 mismatches).
- Episodes: widowx 4 tasks × 50 = 200; fractal 5 × 50 = 250; LIBERO 10 tasks × 10 = 100 per suite (every row, every file).
- `avg_policy_calls > avg_steps`: none. `avg_reuses > steps − calls`: none.
- Duplicated configuration within a file: none (keyed on model_name, env/suite, configuration; also no fully duplicated rows).
- 13-row file: `libero/openvla_libero_10.csv` lacks **`temporal_fusion_conservative_adaptive`** (present in the other 11 LIBERO files and all SIMPLER files). `libero/summary.csv` therefore chose openvla/Libero-Long temporal_fusion from only 2 candidates (motion_entropy 49 % vs task_aware 48 %).
- Zero-success / horizon-saturated rows (avg_steps = horizon; LIBERO horizons 520/300/280/220, widowx mean horizon 75 = (60+60+60+120)/4): smolvla libero_10 depth_pruning4 (0 %, 520 steps), univla libero_goal action_repeat4 (0 %, 300), univla libero_spatial action_repeat4 (0 %, 220), univla libero_object action_repeat4 (1 %, 279.9), univla libero_10 action_repeat4 (1 %, 517.4), cronusvla widowx action_repeat4 (0 %, 75), minivla widowx depth_pruning2 and depth_pruning4 (0 %, 75.0 each).
- Identical-run smells (rows equal in success/steps/calls/reuses): cronusvla widowx guarded strict/moderate/aggressive (36/49.735/49.71/0.025 ×3); cronusvla temporal ×3 in both envs; CogACT motion_entropy = task_aware in both envs; univla widowx original = guarded_strict = temporal_conservative_adaptive (175/30.21/6.375); univla libero_object original = guarded ×3 = conservative_adaptive (96/145.82/14.98); univla libero_spatial original = guarded ×3 = conservative_adaptive (95/108.31/11.27).
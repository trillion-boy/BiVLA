# Figure and media inventory for the anonymous project page

Inventory date: 2026-09-24. Every PNG listed below was opened and inspected; sizes are pixels (width x height).
"Paper" means the final submitted paper (`scratchpad/video_ppt/final_pdf/paper_final.txt`, cross-checked against the
rendered pages of `experiments/paper/video/assets/paper_final_submitted.pdf`). Paths are repository paths under
`/home/user/BiVLA/` unless stated otherwise. No author names, usernames, institutions or URLs appear in any image;
the only "identifying" strings found anywhere are the matplotlib `Software` tag in PNG/PDF metadata
(`Matplotlib version 3.11.x, https://matplotlib.org/`) and the x264 encoder tag in the MP4, both harmless.
The clip manifest JSON and `deck_render/clip_manifest.txt` contain absolute source paths of the form
`/home/user/BiVLA/simulation_rollouts/...` (generic user name, no person name); do not copy those files to the page.

## 1. Map of the paper's five figures to files

| Paper figure | Caption (final paper) | Best file(s) | Notes |
|---|---|---|---|
| Fig. 1 (page 1) | Success rates across 6 VLA backbones and 14 configurations in SimplerEnv WidowX; bars follow setting order darker to lighter; dotted line = original; 0 = no successful episode | `experiments/paper/figures_combined/png/teaser_bridge_bars_3x2_rot2.png` (1042x937) + `pdf/teaser_bridge_bars_3x2_rot2.pdf`; video copy `experiments/paper/video/assets/fig1_teaser.png` (1043x938) | Six panels (a) CogACT, (b) CronusVLA, (c) MiniVLA, (d) OpenVLA, (e) SpatialVLA, (f) UniVLA; 14 bars each; rotated family legend on the right. The `_3x2_rot2` file is the latest (16 Sep) and matches the page-1 layout. Script: `figures_combined/make_teaser_bars.py`. |
| Fig. 2 (page 3) | (a) Where each trick enters the control loop (Guarded Reuse gates, VLA Cache patch reuse, Depth Pruning protected blocks, Foveation sharp fovea / blurred periphery, Action Repeat open-loop interval, pipeline row Observation -> Foveation -> Guarded Reuse -> Visual Encoder -> Temporal Fusion -> Depth Pruning -> Intermediate Action -> Redundancy Control -> Output); (b) Evaluation scope (7 VLA, 3 environments, 6 tricks, 14 configurations) | `experiments/paper/video/assets/vla_method_fig2.pdf` (1 page, 1760x845 pt, vector; rendered check in scratchpad `frames/fig2_pdf.png`). Panel crops as PNG: `m_guarded.png` 2198x1425, `m_cache.png` 2321x1425, `m_prune.png` 1380x1425, `m_fov.png` 2193x1395, `m_fusion.png` 1699x1395, `m_repeat.png` 2007x1400, `m_pipeline.png` 5879x307, `m_scope.png` 852x2600 (all in `experiments/paper/video/assets/`) | IMPORTANT: the matplotlib overview `experiments/paper/figures/overview_A3.png` (and its copies `video/assets/fig2_overview.png`, `ov_*.png`) is an EARLIER Fig. 2 draft, not the submitted one. The `m_*.png` crops are raster crops of the submitted PDF (slightly soft edges at 100%, fine at page width). Use the PDF (or an SVG/PNG rendered from it with `pymupdf`, which is installed) for a crisp page image. |
| Fig. 3 (page 4) | Qualitative evaluation: (a) CogACT on WidowX, original vs depth pruning, green/red borders = success/failure, four instructions (eggplant in basket, spoon on cloth, stack green cube on yellow, carrot on plate); (b) scenes from the four LIBERO suites | (a): `experiments/paper/video/assets/fig3a_cogact.png` (3000x1000). (b): no standalone asset; crop page 4 of `paper_final_submitted.pdf` with pymupdf if needed | (a) is a clean high-resolution raster with no text other than instructions and column titles. |
| Fig. 4 (page 5) | Success-speed trade-off on SimplerEnv WidowX; stars = original, colours = trick types, shaded = faster and no worse | `experiments/paper/figures_combined/png/tradeoff_bridge_col.png` (1042x1043, with 15-entry legend) + `pdf/tradeoff_bridge_col.pdf`; legend-free copy `video/assets/fig4_paper_nolegend.png` (1054x864) | 2x3 panels (a) CogACT ... (f) UniVLA, x = speedup per environment step, y = success change (points). All 13 settings present (CronusVLA reuse and UniVLA task-aware fusion reruns of 16 Sep included). Script: `figures_combined/make_single_column.py` (`tradeoff_col`). |
| Fig. 5 (page 7) | Depth pruning (a, b) and action repeat (c, d) on every backbone and environment; 13 lines for the 22 pairs (LIBERO pooled per backbone) | `experiments/paper/figures_combined/png/ablation_depth_repeat_col.png` (1042x1163) + `pdf/ablation_depth_repeat_col.pdf`; `_short` variant (1042x953) is a tighter layout of the same data | (a) success vs layers removed 0/1/2/4, (b) speedup vs layers, (c) success vs repeat 1/2/4, (d) speedup vs repeat; 13-entry legend "Backbone / Env". SmolVLA/LIBERO speedup at 2 and 4 layers is deliberately omitted (those runs used a second software stack, Section IV-A). Script: `make_single_column.py` (`ablation_col`). |

Removed from the paper for space (16 Sep) but fully consistent with it, and therefore the natural first "Additional Results" item:

| Item | File | Notes |
|---|---|---|
| Cross-setting consistency chart | `experiments/paper/project_page/consistency.png` (1036x510, single column) + `consistency.pdf` + caption `consistency_caption.md`; identical to `figures_combined/png/consistency_v3.png`; video copy `video/assets/consistency.png` (1037x511) | Stacked bars per setting: lower / unchanged / higher success than original over the 13 backbone-environment pairs. Counts in the caption file (Foveation 20% 10/0/3 ... Fusion conservative 4/4/5) were re-derived today from `results_corrected` and match exactly. 1036 px wide is a little soft at full page width; the regenerated catalog version (below) is 2146x1136 with the same numbers, or re-run `make_single_column.py`-style code at 600 dpi. |

## 2. Data-currency caveat for the catalog PNGs (read before using anything from `paper_figures/png`)

The catalog under `experiments/datas/plots/paper_figures/` (README, PAPER_SELECTION, manifest, 35 PNG/PDF pairs) was built on
14 Sep, before the 16 Sep reruns that the final paper uses. Three cells differ from the paper-era data
(`experiments/datas/plots/results_corrected`, which is what the paper figures were drawn from):

| Cell | Catalog (14 Sep) | Current results_corrected (paper) |
|---|---|---|
| CronusVLA WidowX guarded reuse moderate | 35.5 %, 88.8 ms/step | 34.5 %, 122.8 ms/step |
| CronusVLA WidowX guarded reuse aggressive | 36.0 %, 88.1 ms/step | 36.5 %, 122.1 ms/step |
| UniVLA WidowX temporal fusion task-aware | 82.0 %, 203.7 ms/step | 83.5 %, 221.7 ms/step |
| OpenVLA LIBERO temporal fusion conservative | 300/400 episodes, "Incomplete coverage" | 400/400, 70.8 % (complete) |

Consequences: catalog PNGs `tradeoff_cronusvla_simplerenv_bridge`, `tradeoff_univla_simplerenv_bridge`,
`tradeoff_openvla_libero` (shows "Conservative (incomplete)"), `success_change_bridge`, `step_speedup_bridge`,
`success_change_libero`, `step_speedup_libero` (both show "Incomplete coverage"), `evaluation_coverage`
(shows "100 missing: conservative fusion"), `cross_setting_consistency` (12 instead of 13 in one row), all
`ablation_*` (contain the changed cells) and `episode_duration_ecdf_bridge` are STALE. Do not use them as is.

Fix: the whole catalog was regenerated today from the current export into
`scratchpad/website_data/regenerated_catalog/{png,pdf}/` (35 figures, 300 dpi PNG 2146 px wide, vector PDF,
Liberation Serif instead of Times New Roman; script `regenerated_catalog/regen_all.py`, nothing in the repo modified).
The regenerated cross-setting consistency counts equal the project-page caption, and the regenerated audit reports no
incomplete cell. Use the regenerated files, or re-run `paper_figures/scripts/run_all.py` on a machine with Times New Roman
for 600 dpi output (see Section 8).

One more data caveat: the regenerated `ablation_depth_pruning` (and the catalog one) plots SmolVLA/LIBERO speedup at 2 and
4 removed layers (1.006x, 1.113x). The paper's Fig. 5 omits those two points because the runs used a second software
stack (Section IV-A). If this plot goes on the page, either drop those two points or state the caveat under the figure.

## 3. `experiments/paper/figures/` (early Fig. 2 drafts, matplotlib)

| File | Size | Shows | In paper? | Identifying text | Recommendation |
|---|---|---|---|---|---|
| overview_A3.png / .pdf | 2107x906 | Control-loop diagram: pipeline row Observation -> Visual encoder -> Decoder layers -> Output stage -> Action -> Environment, with callout boxes Guarded reuse (gates diamond), Temporal fusion (previous/new/fused grid), Depth pruning (layer stack), Foveation (input/foveated thumbnails of the eggplant scene), Action repeat (call/hold/hold). Grey boxes for controls, colours for candidates | No; superseded by the submitted Fig. 2 (`vla_method_fig2.pdf`). The wording ("Candidates / Controls") predates the final paper, which calls all five "tricks" | None | Skip on the page (or use only as a schematic with its own caption; prefer the submitted Fig. 2) |
| overview_A.png, overview_A2.png | 2148x1005 / 2148x1035 | Same diagram, earlier layout with a "Protocol" strip (Reference, Matched episodes, Grid, Positive gate) and "control/candidate" labels | No | None | Skip |
| overview_B.png, overview_C.png | 2148x900 / 2148x1035 | Further layout variants of the same draft | No | None | Skip |
| tablesimpler_preview.png | 2592x1779 | Rendered preview of an older Table I (WidowX and Google Fractal; Success, Latency, Avg. Steps for CogACT, OpenVLA, SpatialVLA, CronusVLA, UniVLA, MiniVLA at the best setting per trick) | No; preview of a draft table. Several cells predate the 16 Sep reruns (e.g. UniVLA WidowX temporal fusion shown as 87.5 (+0.0)); MiniVLA row is present, CronusVLA latency for reuse is old | None | Skip; take table numbers from the final paper text or `paired_results_all.csv`, never from this image |

## 4. `experiments/paper/figures_combined/png/` (the paper's own Section IV figures and candidates, 300 dpi)

| File | Size | Shows | In paper? | Recommendation |
|---|---|---|---|---|
| teaser_bridge_bars_3x2_rot2.png | 1042x937 | Fig. 1 (see Section 1) | Fig. 1 | Use as is (or its PDF); it is the paper's figure |
| teaser_bridge_bars_3x2_rot.png, _rl.png, _3x2.png (1042x1067), teaser_bridge_bars.png (1042x827, legend on top) | ~1042x937 | Same six bar panels, legend placement variants | Fig. 1 variants | Skip (duplicates) |
| tradeoff_bridge_col.png | 1042x1043 | Fig. 4 (see Section 1) | Fig. 4 | Use as is |
| tradeoff_bridge_row.png | 2226x747 | Same six WidowX trade-off panels in one row, legend below (an earlier full-width Fig. 4 candidate); video copy `video/assets/fig4_tradeoff_row.png` 2196x748 | Draft | Optional alternative layout; same data |
| ablation_depth_repeat_col.png / _short.png | 1042x1163 / 1042x953 | Fig. 5 (see Section 1) | Fig. 5 | Use as is |
| cand_ablation_depth_repeat_row.png | 2146x808 | Same four panels in a row (full-width draft) | Draft | Optional |
| consistency_v3.png (= project_page/consistency.png), consistency_v2.png, cross_setting_consistency_col_short.png (1042x606), cand_cross_setting_consistency_col.png (1048x747) | ~1040x510-750 | Cross-setting consistency stacked bars; v3 is the final single-column version (13 pairs, current data). `col_short` and `cand_` are earlier layouts (cand_ carries the same 13-pair counts, ordering Foveation 20% first) | Removed from paper | Use `project_page/consistency.png` (or the regenerated 2146 px version) |
| cand_success_change_bridge_row.png | 2146x688 | Success change from original, six WidowX backbone panels in a row (CogACT, CronusVLA, MiniVLA, OpenVLA, SpatialVLA, UniVLA), 13 settings each, x-axis label "Success change from dense (percentage points)" | Draft (a `main.tex` figure that was replaced by the Fig. 1 bars) | Usable as an extra only after checking it against `paired_results_all.csv`; note the "dense" wording differs from the paper's "original" |

Scripts in this folder: `make_teaser_bars.py` (Fig. 1), `make_single_column.py` (Fig. 4 `tradeoff_col`, Fig. 5 `ablation_col`),
`make_tradeoff_row.py`, `make_combined_figures.py` (row candidates and consistency). All import the catalog's `common.py`
and use Liberation Serif; they write to `figures_combined/` (would modify the repo if run there) and were not executed.

## 5. `experiments/datas/plots/paper_figures/png/` (35-figure catalog, 600 dpi, 4293 px wide) and the regenerated copies

All 35 are titled-less (identities in captions); README.md and figure_manifest.json hold draft captions;
PAPER_SELECTION.md recommends a six-figure set. Per-figure content (regenerated versions in `website_data/regenerated_catalog/png/` are current; the
repo PNGs are stale where noted in Section 2):

Trade-off plots (2x3 panels: overview with observed frontier, then foveation, action repeat, depth pruning, guarded reuse, temporal fusion; x = time-per-step speedup, y = success change in points; local legends name every setting; light green = faster and higher):

| File (all 4293x3903) | Backbone / environment | Status vs paper | What it shows (current data) | Recommendation |
|---|---|---|---|---|
| tradeoff_cogact_base_simplerenv_bridge | CogACT / WidowX | in Fig. 4 | prune 1-2 layers and fusion above 0; repeat 2/4 at -38/-45 | extra detail; use regenerated |
| tradeoff_cronusvla_simplerenv_bridge | CronusVLA / WidowX | in Fig. 4, catalog STALE | reuse settings +0.5/+2.5 near 1x | use regenerated only |
| tradeoff_minivla_simplerenv | MiniVLA / WidowX | in Fig. 4 | prune 2/4 layers at -36; reuse/fusion +2..+3 | use regenerated |
| tradeoff_openvla_simplerenv | OpenVLA / WidowX | in Fig. 4 | task-aware fusion +7; foveation -13/-20 | use regenerated |
| tradeoff_spatialvla_simplerenv_bridge | SpatialVLA / WidowX | in Fig. 4 | keep 50% +5; repeat 4 at 3.07x, -16 | use regenerated |
| tradeoff_univla_simplerenv_bridge | UniVLA / WidowX | in Fig. 4, catalog STALE (task-aware fusion) | repeat 2/4 at -75/-84.5; everything else within -15..0 | use regenerated only |
| tradeoff_cogact_base_simplerenv_fractal | CogACT / Google robot (Fractal) | NOT in paper figures | all settings within -19..+1; repeat 4 = 1.73x | PRIORITY: use regenerated |
| tradeoff_cronusvla_simplerenv_fractal | CronusVLA / Fractal | NOT in paper figures | all within -10..+1.5; reuse moderate/aggressive slightly above 0 | PRIORITY: use regenerated |
| tradeoff_openvla_simplerenv_fractal | OpenVLA / Fractal | NOT in paper figures | many settings in the green corner: prune 1/2/4 +2.8/+5.2/+1.2, fusion +6.4/+6.8, repeat 2 +4.4; foveation -11/-28 | PRIORITY: use regenerated |
| tradeoff_spatialvla_simplerenv_fractal | SpatialVLA / Fractal | NOT in paper figures | prune 1/2 +1.6/+2.4, reuse strict/aggressive/moderate +2/+2/+1.5 at up to 1.43x; repeat 4 -29.6 | PRIORITY: use regenerated |
| tradeoff_openvla_libero | OpenVLA / LIBERO | NOT in paper figures, catalog STALE (conservative marked incomplete) | prune 4 layers -29; keep 20% -18; repeat 4 -24.5 at 3.14x; conservative fusion -3.3 | PRIORITY: use regenerated only |
| tradeoff_smolvla_libero | SmolVLA / LIBERO | NOT in paper figures | depth pruning -21/-39/-56; foveation -12/-19; reuse within -3..+0.3 | PRIORITY: use regenerated (add the second-stack caveat for pruning speedups) |
| tradeoff_univla_libero | UniVLA / LIBERO | NOT in paper figures | repeat 2/4 -65/-92; prune 1 +1.8; keep 50% +3.3; everything else within -4..+0.3 | PRIORITY: use regenerated |

Other catalog figures:

| File | Size (repo) | Shows | Status | Recommendation |
|---|---|---|---|---|
| success_change_bridge / _fractal / _libero | 4293x4317 / x4317 / x2157 | Horizontal bars of success change for all 13 settings; panels per backbone (WidowX: CogACT, CronusVLA, MiniVLA, OpenVLA, SpatialVLA, UniVLA; Fractal: CogACT, CronusVLA, OpenVLA, SpatialVLA; LIBERO: OpenVLA, SmolVLA, UniVLA) | Extras (Fractal and LIBERO panels are not in the paper). bridge/libero repo PNGs stale | Use regenerated `_fractal` and `_libero` (the Fractal/LIBERO equivalents of Fig. 1) |
| step_speedup_bridge / _fractal / _libero | same | Lollipop plot of time-per-step speedup (log axis) per setting and backbone | Extras; bridge/libero stale | Use regenerated; complements Fig. 4 for Fractal and LIBERO |
| episode_steps_bridge / _fractal / _libero | same | Change in mean episode steps (%) per setting | Extra (supplement-level) | Optional; caption must say fewer steps is not better on its own |
| ablation_foveation / _action_repeat / _depth_pruning / _guarded_reuse / _temporal_fusion | 4293x2097 | Two panels (success change, speedup) with one line per backbone-environment pair (13 lines) across the settings of one trick | depth/repeat are the paper's Fig. 5 in another layout; foveation, reuse, fusion are extras. Repo versions stale | Use regenerated foveation, guarded reuse, temporal fusion as the "remaining three tricks" companion to Fig. 5 |
| libero_suites_openvla / _smolvla / _univla | 4293x2513 | Per-suite success (Spatial, Object, Goal, Long) for all 14 configurations, one backbone each | Extras (Table II in dot form) | Use regenerated; the numbers are Table II, so check labels against the paper |
| episode_duration_ecdf_bridge / _fractal / _libero | 4293x1613 | ECDF of episode duration normalised by the original mean for O, F20, A2, D1, RM, FT, one panel per backbone | Extra | Optional (Additional Discussions); bridge stale |
| cross_setting_consistency | 4293x2273 | Same chart as the project-page consistency figure, full-width | Stale in repo (12/13 row) | Use regenerated (2146x1136) or the project-page single-column file |
| evaluation_coverage | 4293x2453 | Recorded episodes per backbone-environment (2800 WidowX, 3500 Fractal, 5600 LIBERO) | Repo version shows "100 missing" (stale) | Use regenerated (all bars complete) in Implementation Details |
| qa/*.png, qa/overview_*.png | various | QA thumbnails of the same figures | - | Skip |

## 6. `experiments/figures/` (earlier-study material, foveation examples)

| File | Size | Shows | In paper? | Identifying | Recommendation |
|---|---|---|---|---|---|
| obs_carrot_raw / obs_eggplant_raw / obs_spoon_raw / obs_stack_raw.png | 640x480 | Raw SimplerEnv WidowX observations: carrot on plate, eggplant in basket (sink scene), spoon on towel, stack cube | Not as figures (the eggplant frame is the source of Fig. 2's foveation thumbnail) | None | Use as "input" images |
| obs_carrot_blur / obs_eggplant_blur / obs_spoon_blur / obs_stack_blur.png | 640x480 | The same frames after the paper's blur foveation (`foveate_image_blur`, sharp central disc, Gaussian ramp sigma 3 to 9 px outside). Measured sharp-disc radius ~150 px, i.e. keep ratio ~0.2, the paper's "Foveation 20%" | Not in paper, but the mechanism is exactly the paper's | None | PRIORITY: use raw/blur pairs (with the caption "keep ratio 0.2") in Implementation Details |
| obs_carrot_blur_keep10.png | 640x480 | Same blur at keep ratio ~0.1 | Not a paper setting | None | Skip (or mark clearly as illustrative only) |
| obs_*_logpolar.png, obs_carrot_logpolar_keep10.png | 640x480 | Log-polar warp foveation (`foveate_image_logpolar`): periphery smeared in a radial pattern | Not in the final paper (log-polar was an earlier variant; final paper foveation is blur) | None | Skip; would contradict the paper's description |
| logpolar_stages.png | 1478x646 | Five-stage pipeline of the log-polar warp at keep 100% and 20% (input, warpPolar, subsample, inverse warp, difference), dark background | Not in paper | None | Skip |
| logpolar_zoom.png | 926x718 | Centre vs periphery crops of the log-polar output at keep 100% with difference maps | Not in paper | None | Skip |
| hook_points.png (+ .svg, .pptx) | 2000x960 | Three-hook diagram of the earlier three-trick study (foveation, action repeat, depth pruning only; "Three of the four conditions attach here") | Not in paper; contradicts the five-trick design | None | Skip |
| fig1_coverage.png (+ .svg) | 2080x2076 | "The same intervention, opposite answers" grid from the earlier study: OpenVLA/SpatialVLA/UniVLA/RoboVLMs on Bridge, 96 matched pairs, numbers like +18.8, -70.8, log-polar foveation | Not in paper; different backbones (RoboVLMs), different numbers, different protocol | None | Skip (conflicts with the final paper) |
| fig3_horizon.png (+ .svg) | 2080x1200 | "Each policy is best near the horizon it was trained to execute", success vs env steps per model call for OpenVLA/SpatialVLA/UniVLA (earlier study) | Not in paper; numbers (15.6 %, 30.2 %, 78.1 %) do not match final Table I | None | Skip |
| bivla_architecture.svg, bivla_pipeline_userstyle_v5.svg | vector | Earlier "BiVLA" architecture with Motion Gaze, Log-Polar Warp, Chunk-Exec; text includes "Ours" | Not in paper | None (checked all text nodes) | Skip; describes a different method |
| robot_icon.png / .svg | 720x780 | Line-art robot arm icon | - | None | Optional decoration |
| make_result_figures.py, render.sh, render_logpolar_border_safe.py | - | Scripts for the above earlier-study figures (render.sh needs headless Chromium; the logpolar script needs cv2, not installed here) | - | - | Not needed |

## 7. Video material

### 7.1 `experiments/paper/video/ICRA27_video_final.mp4`
179.03 s, 1280x720, H.264 High, 30 fps, 282 kb/s video + 59 kb/s AAC mono, 7.8 MB. Sampled frames at 3, 20, 45, 70, 95, 120,
150, 172 s and the 19-slide contact sheet (`deck_render/contact_sheet_subs.png`, 1920x2400) were inspected. Content
(19 slides, narration burned in as a subtitle band): 1 motivation (Fig. 2 pipeline row); 2-6 one slide per trick with a
Fig. 2 panel and a paired clip in which the original fails and the trick succeeds (foveation 20% CogACT WidowX eggplant
rollout 5; action repeat 4 OpenVLA Fractal pick coke can rollout 1; depth pruning 1 CogACT WidowX stack cube rollout 3;
reuse moderate CogACT WidowX carrot on plate rollout 3; task-aware fusion OpenVLA Fractal close drawer rollout 2);
7-16 "walls" of all 14 configurations on one episode (CogACT WidowX: eggplant, stack cube, spoon on towel; OpenVLA
Fractal: move near, pick coke can, open drawer; UniVLA LIBERO: Spatial task 3, Object task 9, Goal task 8, Long task 1);
17 Fig. 1 with the Table I/II deltas; 18 Fig. 4 with a latency table; 19 takeaways. Title on slide 1 is the paper
title only; "Anonymous" nowhere needed; no names, logos, paths or URLs seen in any sampled frame or in the pptx XML.
Recommendation: embed as is (the page can host the MP4 directly; 7.8 MB). The numbers spoken on slides 17-18 are the
paper's Table I/II best-setting deltas (e.g. CogACT WidowX depth pruning 50.0 -> 59.5; SpatialVLA WidowX 423.5 -> 235.1 ms).
Caveat worth stating under the video: for 5 of the 150 wall tiles (all CogACT WidowX, listed below) the SUCCESS/FAILURE
badge in the rollout video disagrees with `episodes.jsonl` (the WidowX rollout videos are a re-run of the original
experiment); the video follows the badge.

### 7.2 `experiments/paper/video/clips/` (150 MP4 + 150 first-frame PNG + `manifest.json`, 23 MB)
Encoded from `simulation_rollouts/` by `select_clips.py`: 10 fps, no audio, H.264 baseline, played at 2x-8x
(`speed` in manifest), cut to the slide length. Sizes: WidowX 480x360, Fractal 224x288, LIBERO 256x302. Each tile has a
"Step N" counter and a SUCCESS (green) / FAILURE (red) badge burned in. Naming `<env>_<task>_k<rollout>_<config>.mp4`
(walls, 14 configs x 10 episodes) and `pair_<trick>_{orig,trick}.mp4` (5 pairs). `manifest.json` gives per tile the
source path (`/home/user/BiVLA/simulation_rollouts/...`, generic; strip before publishing), success flag, duration and
config. Wall episodes and their success counts (from the badge): CogACT WidowX eggplant rollout 3 (12/14, action repeat
fails), stack cube rollout 5 (11/14), spoon on towel rollout 4 (6/14, original fails); OpenVLA Fractal move near rollout 4
(12/14, foveation fails), pick coke can rollout 4 (13/14), open drawer rollout 5 (8/14, original fails); UniVLA LIBERO
Spatial 3 (12/14), Object 9 (12/14), Goal 8 (12/14), Long 1 (12/14; action repeat fails in all four).
Badge vs record mismatches (from `deck_render/clip_manifest.txt`): eggplant rollout 3 depth_pruning2 (video success,
record failure); stack cube rollout 5 fixed_foveation_keep20 (video success, record failure) and fixed_foveation_keep50
(video failure, record success); pair_prune trick and pair_reuse trick (video success, record failure).
Recommendation: use the wall clips and the five pairs as is (small, anonymous, already sped up). State the episode,
playback speed and the badge caveat in the caption. The 224x288 Fractal tiles are low resolution; fine at thumbnail size only.

### 7.3 `experiments/paper/video/assets/`
| File | Size | Shows | Recommendation |
|---|---|---|---|
| vla_method_fig2.pdf, m_*.png | see Section 1 | Submitted Fig. 2 and its six method panels + pipeline row + scope column | Use (Fig. 2 and per-trick panels in Implementation Details) |
| fig3a_cogact.png | 3000x1000 | Fig. 3(a) | Use |
| fig1_teaser.png, fig1_teaser_nolegend.png | 1043x938 / 945x938 | Fig. 1 with / without legend | Use `figures_combined` originals instead |
| fig4_paper_nolegend.png, fig4_tradeoff_row.png | 1054x864 / 2196x748 | Fig. 4 without legend; row variant | Use `figures_combined` originals instead |
| fig2_overview.png, ov_loop.png (2050x296), ov_fov / ov_repeat (672x222), ov_fusion / ov_guarded / ov_prune (672x276) | - | The superseded matplotlib overview and its crops | Skip |
| consistency.png | 1037x511 | Consistency chart (same as project_page) | Duplicate |
| frame_raw.png, frame_fov20.png, frame_fov50.png | 292x206 | Eggplant scene raw and blurred at keep 20% / 50% (tiny) | Skip; use the 640x480 `obs_eggplant_*` files, or regenerate keep 50% from `obs_eggplant_raw.png` with `foveate_image_blur(img, 0.5)` (needs cv2) |
| paper_final_submitted.pdf | 8 pages | The submitted paper | Source for cropping Fig. 3(b) only |

## 8. Regeneration scripts and whether they run

Location: `experiments/datas/plots/paper_figures/scripts/` (39 files). `common.py` loads every `summary.json` +
`episodes.jsonl` under `experiments/datas/plots/results_corrected/` (13 backbone-environment folders; LIBERO has one
sub-folder per suite), asserts counts, pools episodes, and provides `comparison(name, env, key)`, `pareto(name, model)`,
`ablation(name, family)`, `suite_success(name, model)`, `duration_ecdf(name, env)`, `consistency(name)`, `coverage(name)`.
`run_all.py` regenerates all 35 from `figure_manifest.json`; each `plot_*.py` regenerates one; `build_catalog.py`
rebuilds README/manifest; `verify_outputs.py` checks outputs; `plot_cross_setting_consistency_complete.py` is a
patched variant (now unnecessary, the export is complete).

Per-backbone coverage: trade-off scripts exist for all 13 backbone-environment pairs (`plot_tradeoff_<model>_<env>.py`);
comparison scripts per environment (`plot_success_change_*`, `plot_step_speedup_*`, `plot_episode_steps_*`,
`plot_episode_duration_ecdf_*`); per-LIBERO-backbone suite plots; ablation per trick. Any backbone can also be plotted by
calling `common.pareto('<name>', '<model_env>')` directly.

Run test (no repo modification):
- `python3 plot_tradeoff_openvla_libero.py` as shipped FAILS: `ValueError: Failed to find font Times New Roman ...
  fallback to the default font was disabled` (`common.style()` hard-requires Times New Roman, present only on macOS).
- With `common.style` replaced by a Liberation Serif fallback and `common.OUT` redirected (as the repo's own
  `figures_combined/make_combined_figures.py` does), every function runs: data loads in 1.9 s, each figure 1-2 s,
  all 35 in about 60 s. Dependencies present here: matplotlib 3.11.2, numpy 2.4.6 (requirements pin numpy 2.5.3;
  no incompatibility seen), pillow, scipy. `adjustText` from requirements.txt is not imported by any script.
  Missing: Times New Roman font (only DejaVu and Liberation families installed); `cv2` (needed only by the
  foveation demo scripts, not by the plot scripts); `ffprobe` (the bundled `imageio_ffmpeg` ffmpeg binary works).
- Outputs of the full regeneration: `scratchpad/website_data/regenerated_catalog/{png,pdf}` plus the test run in
  `scratchpad/regen_test/`.

## 9. Suggested page assets, in priority order

1. Additional Results: regenerated trade-off plots for the seven backbone-environment pairs not in Fig. 4
   (CogACT, CronusVLA, OpenVLA, SpatialVLA on Fractal; OpenVLA, SmolVLA, UniVLA on LIBERO); regenerated
   `success_change_fractal`, `success_change_libero`, `step_speedup_{bridge,fractal,libero}`; the cross-setting
   consistency chart (`project_page/consistency.png` + its caption); regenerated `ablation_foveation`,
   `ablation_guarded_reuse`, `ablation_temporal_fusion` (Fig. 5's companions); `libero_suites_*`.
2. Additional Implementation Details: Fig. 2 panels (`m_*.png` or renders from `vla_method_fig2.pdf`); foveation
   examples `obs_{carrot,eggplant,spoon,stack}_{raw,blur}.png` (keep 0.2); regenerated `evaluation_coverage`.
3. Qualitative: `fig3a_cogact.png`; the five paired clips and the ten 14-tile walls from `video/clips/`; the
   accompanying video MP4.
4. Skip: everything under `experiments/figures/` except `obs_*_raw` and `obs_*_blur`; the matplotlib `overview_*`
   drafts and their `ov_*`/`fig2_overview` copies; `tablesimpler_preview.png`; all stale catalog PNGs in the repo;
   `qa/` thumbnails.

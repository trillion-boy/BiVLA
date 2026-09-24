# Additional Discussions: candidate points for the project page

Scope. Each point below is absent from the final paper (Sections IV-B and V, extracted text in paper_final.txt) but is supported by the final per-episode records under `experiments/datas/plots/results_corrected/` (episodes.jsonl and summary.json), by the paired table `experiments/paper/paired_results_all.csv`, or by the code under `adaptive_sparse_vla/`. Every number quoted here was recomputed from those files by `disc_compute.py` in this folder (raw output in `disc_compute_out.txt`). Nothing numeric was taken from Report_EN.md, LIBERO_Report_EN.md or Overview_EN.md, and the older paper notes (Hardware.md, EpisodeCounts.md, CollapseCells.md, BackboneReversal.md, AxisClaim.md, WarpOverhead.md, TableI_Cells.md, PerTaskRows.md) describe an earlier three-backbone study whose numbers do not apply. Where a point was suggested by one of those notes the flag line says so and states what was recomputed.

Conventions. "Per-step latency" is the paper's latency, mean episode wall-clock divided by mean steps. "Per-call latency" is the calls-weighted mean of the per-episode `query_latency_ms.mean_ms` field, which every harness except CronusVLA records. "Gate share" is reused steps divided by all steps. Backbone and environment names follow Tables I and II (Fractal is Google Robot, Long is libero_10). Points are ordered by importance for a reviewer.

---

## 1. Depth pruning flips sign with the layer that Block Influence picks, and the pick sits at a different depth on each backbone

Discussion. The paper reports that the same depth-pruning budget helps CogACT and hurts MiniVLA and SpatialVLA on the same WidowX episodes. The records show what was removed in each case, and the removed index is not comparable across backbones. On the backbones that kept success the lowest-influence block sits in the second half of the decoder, while on the backbones that lost success the criterion selected a block near the protected floor or in the middle of a short stack. The same effect appears inside one backbone on LIBERO, where OpenVLA loses 16 points on the one suite whose calibration frame picked index 8 and 2 to 3 points on the three suites that picked index 17 or 23. This is descriptive, not causal, and the page should say so, but it tells a practitioner that the layer identity has to be reported next to the budget.

Evidence (summary.json `depth_calibration.selected_layers`, one-layer budget unless stated).
- CogACT (32 layers): removes 17 on WidowX, 23 on Fractal. WidowX one and two layers [17], [17, 23]: +7.0 and +9.5 points. Fractal [23], [17, 23], [17, 19, 23, 25]: -1.6, -1.2, 0.0.
- OpenVLA (32): removes 23 on WidowX and Fractal, [23, 25] at two layers. WidowX 0.0, +4.0, -6.0 for one, two, four layers. Fractal +2.8, +5.2, +1.2.
- OpenVLA LIBERO: Long picks 23 (-3.0), Goal 17 (-3.0), Object 17 (-2.0), Spatial 8 (-16.0, p = 0.011). At four layers Spatial removes [8, 20, 23, 25] and drops 30 points.
- UniVLA (32): removes 26 on WidowX (-5.5) and 19 on every LIBERO suite (+2.0, +1.0, +3.0, +1.0).
- MiniVLA (24 layers): removes 13, then [11, 13], then [7, 9, 11, 13]: -17.5, -36.0, -36.0. At two layers and beyond all four WidowX tasks are at 0 of 50.
- SpatialVLA (26 layers): removes 8, then [8, 10], then [8, 10, 13, 23]: WidowX -6.5, -17.0, -31.0. Fractal -5.2, -2.4, -16.0.
- The audit script in `experiments/data_audit_2026-09-11/` reproduces every recorded set with a greedy walk that protects indices below a quarter of the stack, protects the last block and skips neighbours of a removed block, so index 8 of 32 and index 8 of 26 are the first or second eligible index above the floor.
- The per-call saving is close to the removed blocks' share of the decoder on the Llama-style stacks and about half of that on CogACT, whose diffusion action head (10 DDIM steps, cfg 1.5 in `arguments`) is not pruned: four layers cut per-call time by 12.0 percent on OpenVLA WidowX (4 of 32 = 12.5 percent), 12.8 on SpatialVLA WidowX (4 of 26 = 15.4), 12.6 on MiniVLA (4 of 24 = 16.7), 10.5 on UniVLA WidowX, and 5.5 on CogACT WidowX.

Flag. BackboneReversal.md and TableI_Cells.md raise the layer-identity question for the earlier study (a 45.9-point window contrast on OpenVLA Fractal with different runs). None of those numbers is used. The final data have no eligibility-window sweep, so the page must not claim that moving the window changes the sign. The layer sets above are the only layer-identity evidence.

## 2. "Action repeat k" is a different open-loop horizon on a chunking policy, which is why UniVLA collapses at k = 2

Discussion. Action repeat holds each element of the predicted chunk for k steps (`eval.py` line 580, np.repeat semantics, [a, b] becomes [a, a, b, b]). Six backbones emit one action per call, so k = 2 means two environment steps between decisions. UniVLA emits a chunk of 5 actions on WidowX and 10 on LIBERO, so the same nominal setting means 10 and 20 open-loop steps at k = 2 and 20 and 40 at k = 4. The collapse of UniVLA under action repeat is the arithmetic of the horizon, and a table column headed "action repeat" compares three different operations unless the chunk length is printed next to it.

Evidence.
- `policy_calls == ceil(steps / (m k))` holds in 100 percent of the 6,800 action-repeat episodes with m = 1 on every backbone except UniVLA, m = 5 on UniVLA WidowX and m = 10 on UniVLA LIBERO (episodes.jsonl). UniVLA's `checkpoint_manifest.native_action_chunk_size` is 5 on WidowX.
- Success change at k = 2, horizon 2: CogACT -38.0 and -1.2 (WidowX, Fractal), CronusVLA -30.5 and +2.8, OpenVLA -15.5 and +4.4, SpatialVLA -4.0 and 0.0, MiniVLA -6.0, OpenVLA LIBERO -3.0 to -10.0, SmolVLA LIBERO -4.0 to -12.0.
- UniVLA at k = 2, horizon 10 on WidowX: -75.0 (carrot 42 to 12, eggplant 49 to 3, spoon 44 to 6, stack 40 to 4 of 50). Horizon 20 on LIBERO: -59.0, -58.0, -75.0, -68.0. At k = 4 UniVLA scores 3.0 percent on WidowX and 0 to 1 percent on all four LIBERO suites.
- UniVLA's mean policy calls barely fall at k = 2 (WidowX 7.2 against 6.4 for the original) because failed episodes run to the cap and need more calls, so the trick did not even save calls there.

Flag. CollapseCells.md makes this argument for the earlier study with chunk 5 and 96 episodes. All numbers above are from the final records.

## 3. Per-step latency under action repeat is set by the non-model share of a step, and a one-line model predicts every pair within 0.02

Discussion. Action repeat halves or quarters the number of policy calls, but the simulator step, rendering and bookkeeping are paid on every environment step, and the per-call time of the model does not change. The per-step ratio is therefore 1 minus the model's share of a step plus that share divided by k. Backbones whose call is cheap relative to the environment step gain the least. This is the mechanism behind the paper's observation that the speedup is smaller than the call count predicts, and it lets a practitioner predict the gain from the original run alone.

Evidence (full_settings.csv and episodes.jsonl, 20 pairs with per-call latency).
- Model share of a step in the original run (per-call time x calls per step / per-step time): CogACT Fractal 0.59, CogACT WidowX 0.70, MiniVLA 0.65, OpenVLA Fractal 0.67, OpenVLA WidowX 0.76, UniVLA 0.74 to 0.79, SpatialVLA 0.84 and 0.91, OpenVLA LIBERO 0.88 to 0.91, SmolVLA 0.93 to 0.95.
- Predicted against observed per-step ratio at k = 2: CogACT Fractal 0.70 vs 0.72, CogACT WidowX 0.65 vs 0.65, MiniVLA 0.67 vs 0.66, OpenVLA WidowX 0.62 vs 0.62, SpatialVLA WidowX 0.55 vs 0.55, OpenVLA LIBERO 0.54 to 0.56 vs 0.54 to 0.56, SmolVLA 0.52 to 0.53 vs 0.52 to 0.54, UniVLA 0.60 to 0.63 vs 0.59 to 0.64. At k = 4 the largest gap is 0.02 (all 20 pairs).
- Per-call latency under action repeat stays within 1.5 percent of the original on 14 of 20 pairs and within 6.1 percent on all, the rises being CogACT Fractal (2.5 to 4.1 percent) and UniVLA (0.6 to 6.1 percent).

Flag. None. The paper's IV-C draft sentence "the simulator step is paid on every environment step" is the qualitative form of this point.

## 4. Guarded reuse is safe mostly because its gates rarely open, and its speed tracks the gate share exactly

Discussion. The paper calls guarded reuse the most reliable trick. The fire counts show why: on most pairs the gate opens on well under 5 percent of steps, on UniVLA it opens on zero steps in ten of fifteen cells, and the latency change of every cell is the gate share to within a few percent. Where the gate does open often, on OpenVLA LIBERO, the trick saves 8 to 10 percent of the time and costs 1 to 7 points that the paired test does not separate from zero. A reader should therefore treat a green guarded-reuse cell as "did not act" before treating it as "acted safely", and any gated trick should be reported with its fire count.

Evidence (episodes.jsonl `reuses` and `steps_executed`).
- Gate share, strict / moderate / aggressive: OpenVLA LIBERO Long 3.6 / 5.6 / 10.1 percent, Goal 4.8 / 5.1 / 11.2, Object 3.7 / 5.6 / 9.6, Spatial 2.0 / 3.5 / 9.0. OpenVLA WidowX 3.7 / 4.8 / 6.3, Fractal 2.3 / 2.5 / 3.9. MiniVLA 3.1 / 4.2 / 4.0. CronusVLA Fractal 0.7 / 1.7 / 4.2, WidowX 0.05 / 0.3 / 1.4. CogACT WidowX 0.3 / 0.9 / 2.7, Fractal 0.01 / 0.1 / 0.6. SpatialVLA Fractal 0.5 / 0.8 / 2.4, WidowX 0.02 / 0.07 / 0.4. SmolVLA 0.02 to 0.1 / 0.2 to 0.5 / 1.4 to 2.6. UniVLA 0.00 to 0.06 on every cell.
- UniVLA: 0 reuses in Goal, Object and Spatial (all presets) and WidowX strict, 1 to 17 reuses in the other five cells. Success and step count equal the original on every episode except at most two per cell.
- Per-step latency change of the aggressive preset against gate share: OpenVLA Goal -10.4 percent at 11.2, Long -8.9 at 10.1, Object -8.8 at 9.6, Spatial -7.8 at 9.0, OpenVLA WidowX -5.1 at 6.3, MiniVLA -3.1 at 4.0, CronusVLA Fractal -3.1 at 4.2, OpenVLA Fractal -2.1 at 3.9, SpatialVLA Fractal -2.0 at 2.4, CogACT WidowX -1.9 at 2.7.
- paired_results_all.csv: 0 of the 66 guarded-reuse settings changes success at p < 0.05, and 19 of its 22 table cells move by 3 points or less.
- Preset thresholds recorded in `arguments`: strict frame MAE 0.01, patch MAE 0.03, action cosine 0.995, at most 1 consecutive reuse. Moderate 0.015, 0.04, 0.99, 1. Aggressive 0.02, 0.05, 0.98, 2. Minimum translation 0.01 in all three.

Flag. DataAudit_2026-09-11.md finding F6 states this for the September export, and the CronusVLA WidowX moderate and aggressive numbers above are from the 16 September reruns that replaced the runs that note describes.

## 5. Run-to-run noise on WidowX is several points, so small deltas there are only readable through the paired test

Discussion. On a guarded-reuse episode where the gate never fired, the original policy was called at every step, so its outcome should match the original run of the same seed. On SpatialVLA, UniVLA, OpenVLA Fractal and OpenVLA LIBERO it does, on all but at most one episode in a hundred. On CogACT, OpenVLA, CronusVLA and MiniVLA on WidowX the outcome differs on 7 to 23 percent of such episodes. A cell on those four pairs can move by a few points with no trick at all, which is the scale of several green cells in Table I, and the only reading that survives is the paired test with its p-value.

Evidence (episodes.jsonl, zero-fire episodes of the three presets, pooled per pair).
- Outcome differs from the original: CogACT WidowX 86 of 522 (16.5 percent, per preset 10.6 / 23.0 / 16.4), OpenVLA WidowX 55 of 409 (13.4), CronusVLA WidowX 50 of 538 (9.3), MiniVLA 40 of 438 (9.1), CronusVLA Fractal 28 of 609 (4.6), CogACT Fractal 13 of 716 (1.8).
- Reproducing pairs: OpenVLA Fractal 0 of 612, OpenVLA LIBERO 0 of 144, SpatialVLA 0 of 604 and 3 of 572, UniVLA 2 of 300 (Goal) and 0 to 2 elsewhere.
- SmolVLA reuse cells, which ran under the reconstructed stack (point 12): 10.5 to 30.5 percent.
- Table I cells that sit inside this noise band: CogACT WidowX foveation +2.5 (p = 0.60), guarded reuse +3.0 (p = 0.36), CronusVLA WidowX foveation +2.5 (p = 0.44), depth pruning +2.0 (p = 0.59), guarded reuse +2.5 (p = 0.33), MiniVLA guarded reuse +2.5 (p = 0.33), fusion +3.0 (p = 0.39), OpenVLA WidowX guarded reuse +2.0 (p = 0.50). The three CogACT WidowX guarded-reuse presets alone span +1.5, 0.0 and +3.0 with the gate open on 0.3 to 2.7 percent of steps.

Flag. This restates the "run-to-run noise" sentence of the setup.tex draft and the zero_fire.md table in this folder, both recomputed here. The final paper does not print these percentages.

## 6. Twenty-two drops and four gains reach p < 0.05 in the tables, and only the drops survive a multiplicity correction

Discussion. The paper reads its p-values as descriptive because each table cell is the best of two or three settings. Applying the correction the paper leaves out makes the asymmetry explicit. Sixteen of the twenty-two significant drops pass a Benjamini-Hochberg correction at 0.05 over the 110 cells and no gain does, and across all 286 settings the six gains that reach p < 0.05 are about what chance produces. The honest reading of a green success cell is "not worse", and the honest reading of a red one is usually "worse".

Evidence (paired_results_all.csv, exact two-sided McNemar on flipped episodes).
- 110 table cells: 22 lower success at p < 0.05, 4 raise it (CogACT WidowX depth pruning +9.5 at p = 0.009, OpenVLA Fractal task-aware fusion +6.8 at p = 0.012, CogACT WidowX motion-entropy fusion +8.0 at p = 0.020, UniVLA Goal foveation +7.0 at p = 0.039). No green cell exceeds 10 points.
- Benjamini-Hochberg at q = 0.05 over 110 cells: 16 passes, all drops, cut-off p = 0.0061. The four gains fail.
- All 286 settings: 6 gains and 77 drops at p < 0.05. The sum of the exact test sizes at the observed flip counts is 8.0 chance passes, 4.0 per direction. Over the 110 cells it is 3.1.
- Three of the four table gains and the two extra settings that pass (CogACT WidowX one-layer depth +7.0, CogACT WidowX task-aware fusion +8.0) all sit on CogACT WidowX and OpenVLA Fractal, the two pairs on which every depth and fusion setting is positive.

Flag. Numbers agree with fdr.md in this folder. The page should keep the paper's wording that the p-values are descriptive and present this as the quantified form of it.

## 7. The pooled numbers hide a per-task split, and most losses are carried by one or two tasks

Discussion. Success in Tables I and II is pooled over four, five or ten tasks. The per-task counts show that most significant drops come from one or two tasks while the others hold, and that several "unchanged" cells are a gain on one task cancelled by a loss on another. This matters for a practitioner because the failing task is usually the one with the finest contact or the most off-centre target, and because a pooled zero can be two real effects.

Evidence (episodes.jsonl per task, successes of 50 on SimplerEnv and of 10 on LIBERO).
- OpenVLA foveation, WidowX keep 50 (-13.5): eggplant 37 to 12 while carrot 16 to 17 and spoon 19 to 24. Fractal keep 20 (-28.0): move near 31 to 3, pick coke can 29 to 1, open drawer 9 to 0, while close drawer 20 to 15. LIBERO Spatial keep 50 (+5.0): tasks 0 to 3 gain 6 episodes and tasks 6 and 7 one each, while tasks 4, 8 and 9 lose one each.
- MiniVLA one-layer depth pruning (-17.5): stack cube 36 to 8 carries 28 of the 35 lost episodes, spoon 25 to 19, eggplant 9 to 10.
- SpatialVLA WidowX depth pruning: four layers (-31.0) is eggplant 50 to 8 with carrot unchanged at 13, and two layers (-17.0) takes spoon from 11 to 0.
- CogACT WidowX two-layer depth pruning (+9.5) is spread over all four tasks: carrot +8, eggplant +5, spoon +1, stack +5. CogACT Fractal action repeat at k = 4 (-18.8) raises close drawer 31 to 44 while move near falls 47 to 25 and open drawer 40 to 21.
- SmolVLA depth pruning at four layers on Goal (-52.0) leaves tasks 4 and 7 at 10 of 10 while six tasks fall to 0 or 1 and the other two to 2 and 5. On Long the same setting leaves every task at 0.
- UniVLA foveation keep 20 on Long (0.0) is task 8 falling 6 to 2 and task 2 losing two, against task 9 rising 7 to 10 and tasks 1, 4 and 5 gaining one each.

Flag. PerTaskRows.md argues a pick-coke-can versus move-near split from the earlier study. The final Fractal data do not support that split as stated (OpenVLA foveation keep 20 collapses both tasks, depth pruning moves both by a few episodes), so the page should not repeat it. The per-task split above is what the final records show.

## 8. Foveation costs time instead of saving it, and the cost sits outside the model call

Discussion. Foveation blurs the periphery but leaves the image size and the visual token count unchanged, so the model does the same work. The per-call latency of every foveated run is within a few percent of the original, while the per-step latency rises by 3 to 17 percent on six backbones. The difference is the image path, which in the released code is two Gaussian blurs (sigma 3 and 9) and a per-pixel blend on the CPU for every frame. UniVLA pays almost nothing because it blurs one frame per chunk of 5 or 10 steps. Any benefit of foveation is therefore a robustness or stability benefit, and the page should not present it as an efficiency trick.

Evidence (full_settings.csv, per-step and per-call latency, keep 20 / keep 50).
- Per-step change: CronusVLA WidowX +15.9 / +16.8 percent, Fractal +14.9 / +16.0. CogACT WidowX +12.9 / +15.0, Fractal +12.3 / +15.1. MiniVLA +7.2 / +4.6. SpatialVLA WidowX +5.3 / +5.2, Fractal +5.6 / +6.3. SmolVLA +4.2 to +7.4. OpenVLA +2.0 to +4.1 on all seven of its pairs. UniVLA -0.7 to +3.2.
- Added milliseconds per step: SpatialVLA +22 to +28, CronusVLA +20 to +25, CogACT +18 to +26, SmolVLA +8 to +22, MiniVLA +7 to +11, OpenVLA +4 to +10, UniVLA -1 to +2.
- Per-call change on the same runs: CogACT -0.6 to +2.1 percent, SpatialVLA -0.8 to +0.5, OpenVLA -1.1 to +4.8, MiniVLA +0.7 to +5.8, SmolVLA -2.2 to +2.6, UniVLA -1.9 to +3.7.
- Code: `adaptive_sparse_vla/foveation.py` lines 128 to 143 build the foveated frame with `cv2.GaussianBlur(frame, (0, 0), sigmaX=3.0)` and `sigmaX=9.0` and a float32 blend, returning a uint8 image of the same size. The `fusion.representation` and token fields of the fusion runs confirm 256 visual tokens on OpenVLA whether or not the frame is foveated.

Flag. WarpOverhead.md timed a log-polar warp for the earlier study on a different machine. Not used. The final trick is the blur mixture of Eq. (2), and the cost above is read from the run records.

## 9. More blur is not uniformly worse: keep 20 beats keep 50 on four of six WidowX backbones and loses everywhere else

Discussion. The paper's Tables show only the better keep ratio. Across the two settings the ordering is environment dependent. On Fractal and LIBERO the milder keep 50 is at least as good on 15 of 16 pairs, while on WidowX the harsher keep 20 ties one backbone and leads three. The keep ratio is therefore not a monotone knob whose safe direction is known in advance, and a practitioner has to sweep it on the target environment.

Evidence (full_settings.csv, success percent keep 20 vs keep 50).
- WidowX: CogACT 52.5 vs 52.5, CronusVLA 36.5 vs 34.0, SpatialVLA 50.0 vs 44.5, UniVLA 79.0 vs 73.0 (keep 20 ties or leads), MiniVLA 28.0 vs 35.0, OpenVLA 22.5 vs 30.0 (keep 50 leads).
- Fractal: CogACT 64.8 vs 67.2, CronusVLA 54.4 vs 56.8, OpenVLA 7.6 vs 24.4, SpatialVLA 55.2 vs 57.2.
- LIBERO: OpenVLA keep 50 leads on all four suites (Long 30 vs 44, Goal 64 vs 75, Object 74 vs 86, Spatial 55 vs 83), SmolVLA on all four (19 vs 21, 58 vs 67, 74 vs 81, 63 vs 76), UniVLA on three (86 vs 90, 87 vs 99, 91 vs 96) with Object the exception (100 vs 97).

Flag. None.

## 10. Several table cells are the original policy under another name, so a zero change can mean "did not engage"

Discussion. Three fusion configurations never engaged or were duplicates, and the tables cannot show it. Conservative-adaptive fusion reused no patch on SpatialVLA and UniVLA, so those cells are the original run. CogACT exposes no text-to-vision attention where fusion runs, so its task-aware setting is motion-entropy episode for episode. CronusVLA's three fusion settings are identical to each other and their parameters are not recorded. The page should give the engagement counters (keyframe fraction, reusable tokens, fire count) next to every gated or masked trick, because a zero delta and an inert trick look the same in a table.

Evidence (summary.json `fusion` block, episodes.jsonl).
- Conservative-adaptive: `median_reusable_visual_tokens` 0 and keyframes equal to policy calls (fraction 1.00) on all five UniVLA pairs and 0.98 to 0.99 on SpatialVLA. Outcome and steps identical to the original on 200 of 200 episodes (UniVLA WidowX), 100 of 100 (UniVLA Object and Spatial), 98 (Goal), 94 (Long), 247 of 250 (SpatialVLA Fractal), 174 of 200 (SpatialVLA WidowX). On OpenVLA, CogACT and MiniVLA the same setting reuses 64 tokens per call (the 0.25 cap of 256) with keyframe fractions 0.82 to 0.93.
- Table cells affected: UniVLA WidowX temporal fusion (87.50, +0.00) in Table I, UniVLA Object (96.00, +0.00) and Spatial (95.00, +0.00) in Table II.
- CogACT: `task_relevance_supported` false in every file, task-aware identical to motion-entropy in success and steps on 200 of 200 (WidowX) and 250 of 250 (Fractal) episodes. On every other backbone task-aware differs from motion-entropy on at least half of the episodes.
- CronusVLA: no fusion argument in any of its files, `fusion` block empty, and the three settings identical on 200 of 200 and 250 of 250 episodes.
- Guarded reuse: the ten UniVLA cells of point 4 with zero reuses.

Flag. DataAudit_2026-09-11.md findings F2, F3 and F4 record these for the September export. All counts above are from results_corrected after the September reruns.

## 11. Task-aware fusion buys its attention signal with 7 to 17 percent more time per call

Discussion. The task-aware setting adds the text-to-vision attention of the preceding call to the protection mask, which means the attention weights have to be materialised. Every backbone that collects them pays for it on every call, with no change in the visual token count, and on the two backbones whose decoders already expose the weights the cost is under 1 percent. This is the only fusion setting with a measurable latency cost, and the paper's Table cells pick it as the best setting on OpenVLA WidowX, OpenVLA Fractal and OpenVLA Object, so a reader who copies those cells should know the price.

Evidence (per-call latency ratio to the original, `arguments.fusion_collect_relevance` true).
- OpenVLA: WidowX +7.6 percent, Fractal +8.2, Long +9.0, Goal +7.3, Object +7.7, Spatial +7.1. MiniVLA +12.9. UniVLA LIBERO +12.0 (Long), +13.1 (Goal), +12.6 (Object), +16.6 (Spatial).
- Motion-entropy on the same pairs: -1.4 to +2.5 percent (CogACT Fractal +4.8 is the one exception). Conservative-adaptive -1.5 to +3.0.
- SpatialVLA task-aware +0.7 and +0.9 percent, CogACT task-aware +0.3 and -0.5 (it does not collect attention, point 10).
- The UniVLA WidowX task-aware row was rerun separately (its records carry a GPU tag the original row lacks), so its +21 percent is not read.
- Code: `adaptive_sparse_vla/inference.py` line 796 loads the SDPA attention path and `inference_openvla_libero.py` line 122 defaults to eager, which is the backend that returns attention weights. OpenVLA pays the same 7 to 9 percent on LIBERO, where the original already runs eager, so on OpenVLA the cost is the collection itself rather than a backend switch.

Flag. The setup.tex draft states the eager-attention mechanism in one sentence. The per-call numbers are recomputed here.

## 12. The SmolVLA reconstructed-stack cells cannot support a latency reading, and its depth cells are not the paper's depth trick

Discussion. Twenty-nine SmolVLA runs, all its guarded reuse and temporal fusion cells and its two- and four-layer depth cells on Long, Goal and Object, carry the implementation tag `smolvla-reconstructed-sdpa-v1` and GPU tags of up to five card types, while the original and the other cells carry neither. The reconstructed evaluator is 3 to 7 percent faster per call even where the trick does nothing, so every SmolVLA latency change in Table II for those cells is an implementation difference. Its depth cells remove fixed indices without a Block Influence pass, and on the one all-legacy four-layer cell the per-call time does not fall at all while success drops 47 points. The page should state that SmolVLA reuse and fusion cells are read through the paired test alone, that their zero-fire noise is 10 to 36 percent, and that the SmolVLA depth rows measure something other than the decoder saving of Section III-D.

Evidence (episodes.jsonl `implementation` and `gpu` fields, summary.json `depth_calibration`).
- Reconstructed episodes: guarded reuse and temporal fusion 100 of 100 on every suite (RTX 5090), depth pruning two layers 17 of 100 on Long and 26 on Goal, four layers 62 on Long, 100 on Goal and 100 on Object, spread over RTX 5090, RTX PRO 6000, RTX A6000, RTX 6000 Ada and L40S. Depth one layer, Spatial depth cells, foveation and action repeat are legacy.
- Strict guarded reuse fires on 0.02 to 0.13 percent of steps yet its per-call latency is 3.3 to 6.2 percent below the original (Long -5.3, Goal -6.2, Object -5.1, Spatial -3.3), and conservative-adaptive fusion with zero reuses at the summary level is -2.3 to -5.5. These are the stack, not the trick.
- Zero-fire outcome difference from the original: Long 30.5 percent, Goal 23.2, Spatial 21.8, Object 10.5.
- Depth: `calibrated` false, no influence vector, selected indices [30], [28, 30], [24, 26, 28, 30]. Spatial four layers (all legacy): per-call +0.9 percent, success 76 to 29. One layer on all four suites: per-call -0.7 to +1.5 percent, success -14 to -33 points.
- Table II prints latency for all SmolVLA cells. The table generator comment in `tablelibero.tex` intended to print "--" for the reuse and fusion cells.

Flag. DataAudit_2026-09-11.md finding F5 first recorded the two implementations. Counts above are from results_corrected. Note the inconsistency between the paper's Table II, which prints these latencies, and the draft's intent to suppress them, and word the page so that it flags rather than contradicts the table.

## 13. Average steps follows success because every failure runs to the cap

Discussion. Both simulators run a failed episode to its step cap and score at the end. Average steps is therefore a success-weighted mix of the successful episodes' length and the cap, and a change in the Avg. Steps column is mostly a change in success. It is not an independent efficiency measure and should not be read as one.

Evidence (episodes.jsonl).
- 21,568 failed episodes across the 22 pairs, and 21,568 of them end exactly at the task cap (60 or 120 on WidowX, 80 on Fractal, 520, 300, 280 and 220 on LIBERO). Zero failures stop early.
- Examples of the coupling: UniVLA WidowX action repeat k = 2, success 87.5 to 12.5 and steps 30.2 to 71.9. MiniVLA depth pruning at two layers, success 36.0 to 0.0 and steps 63.8 to 75.0, the mean of the four caps. SmolVLA Long depth pruning at four layers, success 42 to 0 and steps 428.4 to 520.0.

Flag. The results.tex draft carried this as one sentence of the Analysis subsection that the final paper dropped.

## 14. What a practitioner should do before applying any of these tricks to a new backbone

Discussion. The findings above reduce to a short protocol. Run the original policy on the exact seeds first, because on some backbones a rerun alone moves a cell by several points. Report the engagement counter of every gated or masked trick, since a zero change can be an inert trick. Print the chunk length and the resulting open-loop horizon next to any action-repeat or reuse setting. Measure per-step wall-clock and per-call time separately, because foveation and task-aware fusion add time outside or inside the call and action repeat saves only the model's share of a step. Record the removed layer indices with the budget, and hold the software stack and GPU class fixed within a comparison.

Evidence. Points 1 to 13. The specific expectations the final data support for a Llama-style 32-layer decoder are one or two removed layers at a per-call saving of 3 percent per layer with success kept on CogACT, OpenVLA SimplerEnv and UniVLA, and a loss on MiniVLA, SpatialVLA and SmolVLA. For a chunking policy action repeat at any k is a horizon of k times the chunk and collapsed on UniVLA. For guarded reuse the strict preset opened on under 1 percent of steps on 15 of 22 pairs.

Flag. None.

---

## The paper's own limitations, to restate on the page

The final paper has no separate limitations paragraph. It says in Section IV-B that "Additional details on evaluation setup, methodology, key assumptions, and limitations are provided on the project page". The statements below are the paper's own caveats, quoted closely from paper_final.txt, followed by the fuller limitation paragraph of the authors' draft conclusion (conclusion.tex, 15 September 2026), each item checked against the final data.

From the final paper.
- Section III-D, depth pruning: "This approximation relies on a distributional assumption: a block that appears redundant on calibration trajectories may still be important in unseen scenes, instructions, or contact states. Moreover, cosine similarity measures representational alignment rather than equivalence in the resulting action."
- Section IV-A, tables: "Tables I and II report the best setting for each trick, chosen by highest success and then fewer average steps."
- Section IV-A, significance: "Since each table entry is selected from multiple settings, the corresponding p-values are used only as descriptive evidence."
- Section IV-A, hardware: "All experiments are run on a shared GPU cluster, with hardware kept consistent within each backbone when comparing latency."
- Section IV-B: "Action repeat gives large speed benefits but is less suitable as a success-preserving method, especially when repeated actions extend the open-loop interval."
- Section IV-B: "This shows that structural redundancy is not a fixed property of the trick alone; it also depends on how the backbone uses its decoder layers during control."
- Section V: "Our main finding was that VLA efficiency was policy- and environment-dependent. ... This made matched evaluation important before applying any efficiency trick to a new VLA."

From the draft conclusion (verbatim-adjacent, each checked).
- "Each cell is one run, so the paired test speaks only to the tested initial states." Checked: one run per configuration in results_corrected, 14 configurations per pair.
- "CogACT, CronusVLA, MiniVLA and OpenVLA do not reproduce their WidowX rollouts under a fixed seed." Checked: point 5, 9 to 17 percent of zero-fire episodes change outcome.
- "LIBERO cells, with 100 episodes, have less power than SimplerEnv cells." Checked: 100 episodes per suite against 200 and 250.
- "Guarded reuse ran fixed presets rather than thresholds tuned per policy." Checked: identical `reuse_*` arguments on every backbone (point 4).
- "Latency is read only within one backbone, environment, software stack and GPU class, peak memory was not measured, and all runs are in simulation." Checked: no memory field in any record, GPU recorded only for the reruns and the SmolVLA reconstructed cells, every environment a simulator.
- "Several seeds per cell, real-robot replication of the sign flips of Section IV-B, and a measured integration of the fusion reuse mask with a token cache are the next steps." Consistent with the paper's Section III-G, which shares the mask with the cache but reports no combined acceleration.

Additional caveats the final data impose (not in either text).
- CogACT tested two fusion settings, not three (point 10). CronusVLA's fusion settings ran with unrecorded parameters (point 10). SmolVLA's depth cells were not calibrated (point 12).
- The SmolVLA latency cells for reuse, fusion and depth two and four on Long, Goal and Object compare two implementations on several GPU classes (point 12).
- Three runs (UniVLA WidowX task-aware, CronusVLA WidowX moderate and aggressive reuse) are reruns whose records carry a GPU tag the original rows lack, so their latency is not compared.

---

## Points from the internal notes that were considered and dropped

- The 45.9-point eligibility-window contrast, the `window875` and `prune4_gap3` controls (BackboneReversal.md, TableI_Cells.md, Report_EN.md). Earlier study, no counterpart in results_corrected.
- The T4 and L4 hardware assignment and the 3.1-point cross-card bound (Hardware.md, EpisodeCounts.md). Earlier study on different hardware. The final cluster records an RTX 5090 where it records anything.
- The "pick coke can holds up better than move near" split (PerTaskRows.md). Not reproduced by the final Fractal records (point 7 flag).
- The missing-EOS slowdown under a deep-end window on SpatialVLA (Report_EN.md 3.5.2). Console observation from the earlier study without result files.
- Log-polar warp timing (WarpOverhead.md). The final trick is the blur mixture, costed in point 8 from the run records.
- The claim that fusion "never lowers success significantly". True for the table cells and for 64 of 66 fusion settings, but OpenVLA Long conservative-adaptive is -11.0 at p = 0.007 and UniVLA WidowX motion-entropy is -5.5 at p = 0.035, so the page should say "in no table cell" rather than "never".

## Hygiene for the page builder

- summary.json `arguments.checkpoint`, `output_dir` and `checkpoint_manifest` fields contain absolute machine paths with user names. Never copy them. Quote only the field names used above.
- The `slurm_job_id`, `recorded_at_utc` and `qos` fields of the SmolVLA and rerun records identify a cluster. Do not print them.
- Refer to the people who ran the study as "the authors" throughout.

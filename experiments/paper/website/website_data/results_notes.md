# Notes on the result data files

All files were built by build_results_data.py in this folder from the per-episode records (episodes.jsonl) under results_corrected, the paired table paired_results_all.csv, and the extracted text of the final paper. Numbers are given to two decimals where the paper's tables use two (success %, latency ms, average steps, changes) and to one decimal for percentages quoted in the paper's text (zero_fire.md); p-values to four decimals.

## Files

- **full_settings.csv** (308 rows = 22 pairs x 14 configurations). One row per backbone x environment (or LIBERO suite) x configuration. `episodes`, `successes`, `avg_steps`, `avg_policy_calls`, `avg_reuses` and `avg_episode_s` are computed from episodes.jsonl (they equal summary.json where that file carries the field). `latency_ms_per_step` = mean episode wall-clock ms / mean steps, the paper's latency definition. `policy_query_latency_mean_ms` is the calls-weighted mean of the per-episode per-call means (the summary.json definition); `inference_cycle_mean_ms` likewise where recorded. `delta_*_vs_original` are differences of the unrounded cell values. `win`, `loss`, `p_mcnemar`, `dsteps` (with 95% bootstrap interval), `dtime_s` and `in_table` are joined from paired_results_all.csv; the original rows carry `in_table` = 1 and empty paired fields. `results_folder` is the source folder relative to results_corrected.
- **per_task.csv**: per backbone x env x configuration x task, with the same quantities for the original run and the per-task delta. `task_description` is the LIBERO task language in the benchmark's task order (index k of `<suite>__task_k`); SimplerEnv task names are the harness names.
- **pairs.csv**: the 22 pairs with episode counts, tasks, step caps (largest steps_executed observed per task over all 14 configurations, which equals the cap named in Sec. IV-A), seed rule, folder layout, which configurations record per-call latency, the GPU where the records carry one, and the implementation tag of the SmolVLA reconstructed-stack episodes.
- **fdr.md**: the false discovery rate check and chance-pass counts of Sec. IV-A/IV-B.
- **zero_fire.md**: run-to-run noise from the zero-fire guarded reuse episodes.

## Name mapping (results_corrected folder -> paired_results_all.csv backbone, env)

| folder | backbone | env |
|---|---|---|
| cogact_base_simplerenv_bridge | CogACT | WidowX |
| cogact_base_simplerenv_fractal | CogACT | Fractal |
| cronusvla_simplerenv_bridge | CronusVLA | WidowX |
| cronusvla_simplerenv_fractal | CronusVLA | Fractal |
| minivla_simplerenv | MiniVLA | WidowX |
| openvla_simplerenv | OpenVLA | WidowX |
| openvla_simplerenv_fractal | OpenVLA | Fractal |
| spatialvla_simplerenv_bridge | SpatialVLA | WidowX |
| spatialvla_simplerenv_fractal | SpatialVLA | Fractal |
| univla_simplerenv_bridge | UniVLA | WidowX |
| openvla_libero/<config>/libero_10, libero_goal, libero_object, libero_spatial | OpenVLA | LIBERO Long, LIBERO Goal, LIBERO Object, LIBERO Spatial |
| smolvla_libero/<config>/libero_10, libero_goal, libero_object, libero_spatial | SmolVLA | LIBERO Long, LIBERO Goal, LIBERO Object, LIBERO Spatial |
| univla_libero/<config>/libero_10, libero_goal, libero_object, libero_spatial | UniVLA | LIBERO Long, LIBERO Goal, LIBERO Object, LIBERO Spatial |

Configuration name -> family: fixed_foveation_keep20/50 = Foveation (keep 20% / 50% of the image area); action_repeat2/4 = Action repeat (k = 2 / 4); depth_pruning1/2/4 = Depth pruning (1 / 2 / 4 layers); guarded_reuse_strict/moderate/aggressive = Guarded reuse; temporal_fusion_motion_entropy / task_aware / conservative_adaptive = Temporal fusion. In the paper's Table I and II the LIBERO suite libero_10 is called Long and Google Robot is called Fractal.

## Folder layouts encountered

- CogACT Fractal: one folder per task inside the configuration folder
- CogACT WidowX: single summary per configuration
- CronusVLA Fractal: one folder per task inside the configuration folder
- CronusVLA WidowX: one folder per task inside the configuration folder
- MiniVLA WidowX: single summary per configuration
- OpenVLA Fractal: one folder per task inside the configuration folder
- OpenVLA LIBERO Goal: single summary per suite
- OpenVLA LIBERO Long: single summary per suite
- OpenVLA LIBERO Object: single summary per suite
- OpenVLA LIBERO Spatial: single summary per suite
- OpenVLA WidowX: one folder per task inside the configuration folder
- SmolVLA LIBERO Goal: single summary per suite
- SmolVLA LIBERO Long: single summary per suite
- SmolVLA LIBERO Object: single summary per suite
- SmolVLA LIBERO Spatial: single summary per suite
- SpatialVLA Fractal: one folder per task inside the configuration folder
- SpatialVLA WidowX: one folder per task inside the configuration folder
- UniVLA LIBERO Goal: single summary per suite
- UniVLA LIBERO Long: single summary per suite
- UniVLA LIBERO Object: single summary per suite
- UniVLA LIBERO Spatial: single summary per suite
- UniVLA WidowX: one folder per task inside the configuration folder

Where a configuration is split into one folder per task, the task folders were concatenated; their summary.json files differ in schema (some CronusVLA per-task summaries carry only episode and success counts, the 2026-09-16 CronusVLA WidowX reuse reruns and the SmolVLA reconstructed-stack cells carry a `gpus` field and a different timing block), so every cell aggregate was recomputed from the episode records and compared with summary.json where a field exists.

## Cross-check of the in-table cells against the paper's Tables I and II

Checked 132 cells (22 original rows + 110 best-setting cells), each on success %, latency ms and average steps, plus the parenthesised change. The paper's latency cell is avg_episode_time_s (rounded to 0.01 s for the cells taken from the per-backbone CSVs, to 1 ms or unrounded for the cells computed from the records) x 1000 / avg_steps, and the check accepts a match under any of the three; `latency_ms_per_step` in full_settings.csv is unrounded and can differ from the printed cell by up to about 0.1 ms on short-episode pairs. The printed change is the raw change rounded to two decimals and can differ by 0.01 from the difference of the two printed numbers.

**No mismatch.** Every printed success, latency, step and change value agrees with the records.

Latency cells matching each rounding of the mean episode time before division by mean steps: rounded 0.01 s episode time: 115; rounded 1 ms episode time: 15; unrounded episode time: 2. The cells not matching the 0.01 s rounding are CronusVLA cells and SmolVLA reconstructed-stack cells, which the table generator computed from the records rather than from the per-backbone CSVs: CronusVLA WidowX original (rounded 1 ms episode time); CronusVLA WidowX fixed_foveation_keep20 (rounded 1 ms episode time); CronusVLA WidowX action_repeat2 (rounded 1 ms episode time); CronusVLA WidowX guarded_reuse_aggressive (unrounded episode time); CronusVLA WidowX temporal_fusion_motion_entropy (rounded 1 ms episode time); CronusVLA Fractal original (rounded 1 ms episode time); CronusVLA Fractal fixed_foveation_keep50 (rounded 1 ms episode time); CronusVLA Fractal action_repeat2 (rounded 1 ms episode time); CronusVLA Fractal depth_pruning1 (rounded 1 ms episode time); CronusVLA Fractal guarded_reuse_aggressive (rounded 1 ms episode time); CronusVLA Fractal temporal_fusion_motion_entropy (rounded 1 ms episode time); SmolVLA LIBERO Long temporal_fusion_conservative_adaptive (rounded 1 ms episode time); SmolVLA LIBERO Goal temporal_fusion_conservative_adaptive (rounded 1 ms episode time); SmolVLA LIBERO Object guarded_reuse_aggressive (rounded 1 ms episode time); SmolVLA LIBERO Object temporal_fusion_task_aware (rounded 1 ms episode time); SmolVLA LIBERO Spatial guarded_reuse_strict (rounded 1 ms episode time); SmolVLA LIBERO Spatial temporal_fusion_task_aware (unrounded episode time). The conventions differ by at most 0.1 ms and never change a printed value by more than its last digit; `latency_ms_per_step` in full_settings.csv is unrounded for every cell.

## Paired table

- win/loss counts, n, McNemar p, dsteps and dtime_s of all 286 rows of paired_results_all.csv were reproduced from the episode records (pairing on task and episode index, no unpaired episodes).
- The `in_table` flag matches the selection rule (highest success, then fewer average steps) on all 110 family cells.
- The Markdown table PairedResultsAll.md in the paper folder predates the 16 September reruns (it lists CronusVLA WidowX guarded reuse strict as the table cell with 9/5 flips); paired_results_all.csv and project_page/pvalues.csv carry the rerun values (aggressive, 11/6) and were used.

## Anomalies and remarks

- OpenVLA LIBERO Object original: summary query mean 157.30 vs episode-derived 157.84 ms.
- OpenVLA LIBERO Spatial original: summary query mean 160.39 vs episode-derived 159.93 ms.
- OpenVLA LIBERO Long depth_pruning1: summary query mean 153.44 vs episode-derived 153.13 ms.
- UniVLA LIBERO Object original: summary query mean 487.89 vs episode-derived 482.34 ms.
- UniVLA LIBERO Spatial original: summary query mean 519.19 vs episode-derived 514.17 ms.
- The summary.json per-call latency of five resumed LIBERO cells (listed above if flagged) was computed by the harness over the resumed portion of the run only, so it differs from the episode-derived calls-weighted mean by 0.3 to 5.5 ms (at most 1.1 percent); full_settings.csv uses the episode-derived value. The tables' latency uses episode wall-clock and is unaffected.
- Per-call latency: CronusVLA's harness records no per-call time except in the two 2026-09-16 WidowX reuse reruns (moderate, aggressive), whose records carry raw timing samples that include the warm-up first call; their `policy_query_latency_mean_ms` is pooled over those samples and the paper does not read it. `inference_cycle_mean_ms` is absent for CogACT and for the CronusVLA cells other than those two reruns.
- GPU: the episode records name the card only for the CronusVLA WidowX moderate and aggressive reruns, the UniVLA WidowX task-aware rerun, and the SmolVLA reconstructed-stack cells (see pairs.csv). These three reruns record an RTX 5090, as the project-page pvalues.md also states; setup.tex's Hardware paragraph mentions an RTX PRO 6000 in the same sentence, which the records do not show for these three runs.
- SmolVLA: guarded reuse and temporal fusion (all episodes), depth_pruning2 on Long and Goal (partly) and depth_pruning4 on Long, Goal and Object ran under the reconstructed SDPA stack on several GPU classes; the paper does not read their latency. The final paper's Table II nonetheless prints latency numbers in those cells, and the cross-check above compares them.
- Internal reports (Report_EN.md, LIBERO_Report_EN.md, Overview_EN.md) were not used for any number.

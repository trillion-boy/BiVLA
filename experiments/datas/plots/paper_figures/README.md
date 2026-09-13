# Paper figure catalog

35 separate figures; each has a Python script, 600-dpi PNG, and vector PDF with embedded Times New Roman.

## Recommended paper selection

See [PAPER_SELECTION.md](PAPER_SELECTION.md) for a concrete six-figure starting set and a LaTeX example.

Use the success-change bar plots and speedup dot plots for the principal benchmarks, 2–3 representative trade-off figures, the relevant configuration sweeps, and the cross-setting consistency plot. Use LIBERO suite plots where task dependence is central. Keep the remaining figures in the supplement; putting every generated figure in the main paper would be repetitive.

## Reproduce

From the original plots folder:

```sh
.venv/bin/python paper_figures/scripts/run_all.py
# Or regenerate one figure:
.venv/bin/python paper_figures/scripts/plot_success_change_libero.py
```

Fresh environment: `python -m venv .venv` then `.venv/bin/python -m pip install -r paper_figures/requirements.txt`. Times New Roman must be installed; the scripts fail explicitly if unavailable. Source paths are resolved relative to scripts, so invocation does not depend on the working directory.

## Metrics and scope

Source of truth: all 699 summaries and their 47,500 episode records in results_corrected. Counts and successes agree for every file. Inputs are untouched. Each model is kept separate by environment. Aggregation pools episodes within a setting; LIBERO has equal-sized suites and the supplied SimplerEnv tasks have equal-sized trial sets. Full comparisons require identical task/episode-ID coverage to original.

Success is 100 × successes / episodes. Effective time per step is sum(episode_elapsed_ms) / sum(steps_executed), including whatever overhead is in the recorded timer. It is not model-query latency. Speedup is original effective time per step / modified effective time per step. Episode steps and durations include failures. Query latency in exported metrics is weighted by policy-call count and is missing if any episode lacks query timing. No missing query timings are inferred from episode durations.

All results are descriptive point estimates from the supplied runs. No error bars or significance claims are fabricated; seed-to-seed replication and confirmed pairing protocols are needed for stronger statistical claims. Matching task/episode IDs is a coverage check, not proof of random-seed pairing. Hardware fields are incomplete in table1; cross-model absolute runtime or parameter-size speed claims are not made. Model names and parameter counts in table1 do not by themselves justify parameter-versus-latency regression. No training curves, memory plots, energy plots, or latency violin plots are generated because the necessary measurements are absent.

## Data limitations and table reconciliation

- openvla_libero/temporal_fusion_conservative_adaptive: 300/400 baseline task/episode IDs; excluded from aggregate comparisons.
- Table2 uses family-level labels that do not specify every exact configuration. `data/table2_config_matches.csv` records candidates that reproduce success and average steps. Figure labels explicitly state configurations.
- Table2 latency is consistent with elapsed time per environment step in inspected baseline rows, rather than model-query latency. These figures recompute effective step time directly from episode logs. Table2 leaves SmolVLA reuse/fusion latency blank, although episode elapsed times are present; the derived step-time plots use those recorded episode durations and do not claim that query timing was measured.
- Do not interpret empirical Pareto selection on this evaluation set as a validated deployment selection procedure.

## Figures and draft captions

### 01. success_change_bridge

Placement: Main paper.

[PNG](png/success_change_bridge.png) · [PDF](pdf/success_change_bridge.pdf) · [Python](scripts/plot_success_change_bridge.py)

Success-rate change relative to each original policy, in percentage points. All 13 non-original configurations are shown in aligned backbone panels with shared axes. Color identifies the intervention family. Bars extend from zero for success and episode-step changes. Missing full-benchmark coverage is labeled explicitly. Backbone panels, left to right and then top to bottom: CogACT, CronusVLA, MiniVLA, OpenVLA, SpatialVLA, UniVLA. Figure and panel titles are omitted; panel identities are specified in this caption and the legends.

### 02. step_speedup_bridge

Placement: Main paper.

[PNG](png/step_speedup_bridge.png) · [PDF](pdf/step_speedup_bridge.pdf) · [Python](scripts/plot_step_speedup_bridge.py)

Effective time-per-step speedup: original pooled elapsed time per step divided by the configuration value. Includes recorded rollout overhead. Dots show measured speedup on a logarithmic axis; stems connect each measurement to the 1× original-policy reference. All 13 non-original configurations are shown in aligned backbone panels with shared axes. Color identifies the intervention family. Bars extend from zero for success and episode-step changes. Missing full-benchmark coverage is labeled explicitly. Backbone panels, left to right and then top to bottom: CogACT, CronusVLA, MiniVLA, OpenVLA, SpatialVLA, UniVLA. Figure and panel titles are omitted; panel identities are specified in this caption and the legends.

### 03. episode_steps_bridge

Placement: Supplement.

[PNG](png/episode_steps_bridge.png) · [PDF](pdf/episode_steps_bridge.pdf) · [Python](scripts/plot_episode_steps_bridge.py)

Change in mean episode steps relative to original, including successful and failed episodes. Fewer steps alone do not establish better policy performance. All 13 non-original configurations are shown in aligned backbone panels with shared axes. Color identifies the intervention family. Bars extend from zero for success and episode-step changes. Missing full-benchmark coverage is labeled explicitly. Backbone panels, left to right and then top to bottom: CogACT, CronusVLA, MiniVLA, OpenVLA, SpatialVLA, UniVLA. Figure and panel titles are omitted; panel identities are specified in this caption and the legends.

### 04. success_change_fractal

Placement: Main paper.

[PNG](png/success_change_fractal.png) · [PDF](pdf/success_change_fractal.pdf) · [Python](scripts/plot_success_change_fractal.py)

Success-rate change relative to each original policy, in percentage points. All 13 non-original configurations are shown in aligned backbone panels with shared axes. Color identifies the intervention family. Bars extend from zero for success and episode-step changes. Missing full-benchmark coverage is labeled explicitly. Backbone panels, left to right and then top to bottom: CogACT, CronusVLA, OpenVLA, SpatialVLA. Figure and panel titles are omitted; panel identities are specified in this caption and the legends.

### 05. step_speedup_fractal

Placement: Main paper.

[PNG](png/step_speedup_fractal.png) · [PDF](pdf/step_speedup_fractal.pdf) · [Python](scripts/plot_step_speedup_fractal.py)

Effective time-per-step speedup: original pooled elapsed time per step divided by the configuration value. Includes recorded rollout overhead. Dots show measured speedup on a logarithmic axis; stems connect each measurement to the 1× original-policy reference. All 13 non-original configurations are shown in aligned backbone panels with shared axes. Color identifies the intervention family. Bars extend from zero for success and episode-step changes. Missing full-benchmark coverage is labeled explicitly. Backbone panels, left to right and then top to bottom: CogACT, CronusVLA, OpenVLA, SpatialVLA. Figure and panel titles are omitted; panel identities are specified in this caption and the legends.

### 06. episode_steps_fractal

Placement: Supplement.

[PNG](png/episode_steps_fractal.png) · [PDF](pdf/episode_steps_fractal.pdf) · [Python](scripts/plot_episode_steps_fractal.py)

Change in mean episode steps relative to original, including successful and failed episodes. Fewer steps alone do not establish better policy performance. All 13 non-original configurations are shown in aligned backbone panels with shared axes. Color identifies the intervention family. Bars extend from zero for success and episode-step changes. Missing full-benchmark coverage is labeled explicitly. Backbone panels, left to right and then top to bottom: CogACT, CronusVLA, OpenVLA, SpatialVLA. Figure and panel titles are omitted; panel identities are specified in this caption and the legends.

### 07. success_change_libero

Placement: Main paper.

[PNG](png/success_change_libero.png) · [PDF](pdf/success_change_libero.pdf) · [Python](scripts/plot_success_change_libero.py)

Success-rate change relative to each original policy, in percentage points. All 13 non-original configurations are shown in aligned backbone panels with shared axes. Color identifies the intervention family. Bars extend from zero for success and episode-step changes. Missing full-benchmark coverage is labeled explicitly. Backbone panels, left to right and then top to bottom: OpenVLA, SmolVLA, UniVLA. Figure and panel titles are omitted; panel identities are specified in this caption and the legends.

### 08. step_speedup_libero

Placement: Main paper.

[PNG](png/step_speedup_libero.png) · [PDF](pdf/step_speedup_libero.pdf) · [Python](scripts/plot_step_speedup_libero.py)

Effective time-per-step speedup: original pooled elapsed time per step divided by the configuration value. Includes recorded rollout overhead. Dots show measured speedup on a logarithmic axis; stems connect each measurement to the 1× original-policy reference. All 13 non-original configurations are shown in aligned backbone panels with shared axes. Color identifies the intervention family. Bars extend from zero for success and episode-step changes. Missing full-benchmark coverage is labeled explicitly. Backbone panels, left to right and then top to bottom: OpenVLA, SmolVLA, UniVLA. Figure and panel titles are omitted; panel identities are specified in this caption and the legends.

### 09. episode_steps_libero

Placement: Supplement.

[PNG](png/episode_steps_libero.png) · [PDF](pdf/episode_steps_libero.pdf) · [Python](scripts/plot_episode_steps_libero.py)

Change in mean episode steps relative to original, including successful and failed episodes. Fewer steps alone do not establish better policy performance. All 13 non-original configurations are shown in aligned backbone panels with shared axes. Color identifies the intervention family. Bars extend from zero for success and episode-step changes. Missing full-benchmark coverage is labeled explicitly. Backbone panels, left to right and then top to bottom: OpenVLA, SmolVLA, UniVLA. Figure and panel titles are omitted; panel identities are specified in this caption and the legends.

### 10. tradeoff_cogact_base_simplerenv_bridge

Placement: Main paper: select representative settings; remaining plots in supplement.

[PNG](png/tradeoff_cogact_base_simplerenv_bridge.png) · [PDF](pdf/tradeoff_cogact_base_simplerenv_bridge.pdf) · [Python](scripts/plot_tradeoff_cogact_base_simplerenv_bridge.py)

Success change in percentage points versus effective time-per-step speedup, relative to the original policy in the same model and environment. Panels in reading order (left to right, top to bottom): all configurations, foveation, action repeat, depth pruning, guarded reuse, and temporal fusion. Family panels use independent axis limits. Black stars and dotted reference lines mark the original policy (1×, 0 pp). Light green marks simultaneous improvement in success and speed. Local legends identify every configuration; colors encode families, and marker shapes distinguish settings within a family. The dashed line in the overview connects empirically nondominated measurements and does not imply interpolated or statistically significant performance. Incomplete full-benchmark measurements are omitted and labeled in the relevant legend. Figure and panel titles are omitted; panel identities are specified in this caption and the legends.

### 11. tradeoff_cogact_base_simplerenv_fractal

Placement: Main paper: select representative settings; remaining plots in supplement.

[PNG](png/tradeoff_cogact_base_simplerenv_fractal.png) · [PDF](pdf/tradeoff_cogact_base_simplerenv_fractal.pdf) · [Python](scripts/plot_tradeoff_cogact_base_simplerenv_fractal.py)

Success change in percentage points versus effective time-per-step speedup, relative to the original policy in the same model and environment. Panels in reading order (left to right, top to bottom): all configurations, foveation, action repeat, depth pruning, guarded reuse, and temporal fusion. Family panels use independent axis limits. Black stars and dotted reference lines mark the original policy (1×, 0 pp). Light green marks simultaneous improvement in success and speed. Local legends identify every configuration; colors encode families, and marker shapes distinguish settings within a family. The dashed line in the overview connects empirically nondominated measurements and does not imply interpolated or statistically significant performance. Incomplete full-benchmark measurements are omitted and labeled in the relevant legend. Figure and panel titles are omitted; panel identities are specified in this caption and the legends.

### 12. tradeoff_cronusvla_simplerenv_bridge

Placement: Main paper: select representative settings; remaining plots in supplement.

[PNG](png/tradeoff_cronusvla_simplerenv_bridge.png) · [PDF](pdf/tradeoff_cronusvla_simplerenv_bridge.pdf) · [Python](scripts/plot_tradeoff_cronusvla_simplerenv_bridge.py)

Success change in percentage points versus effective time-per-step speedup, relative to the original policy in the same model and environment. Panels in reading order (left to right, top to bottom): all configurations, foveation, action repeat, depth pruning, guarded reuse, and temporal fusion. Family panels use independent axis limits. Black stars and dotted reference lines mark the original policy (1×, 0 pp). Light green marks simultaneous improvement in success and speed. Local legends identify every configuration; colors encode families, and marker shapes distinguish settings within a family. The dashed line in the overview connects empirically nondominated measurements and does not imply interpolated or statistically significant performance. Incomplete full-benchmark measurements are omitted and labeled in the relevant legend. Figure and panel titles are omitted; panel identities are specified in this caption and the legends.

### 13. tradeoff_cronusvla_simplerenv_fractal

Placement: Main paper: select representative settings; remaining plots in supplement.

[PNG](png/tradeoff_cronusvla_simplerenv_fractal.png) · [PDF](pdf/tradeoff_cronusvla_simplerenv_fractal.pdf) · [Python](scripts/plot_tradeoff_cronusvla_simplerenv_fractal.py)

Success change in percentage points versus effective time-per-step speedup, relative to the original policy in the same model and environment. Panels in reading order (left to right, top to bottom): all configurations, foveation, action repeat, depth pruning, guarded reuse, and temporal fusion. Family panels use independent axis limits. Black stars and dotted reference lines mark the original policy (1×, 0 pp). Light green marks simultaneous improvement in success and speed. Local legends identify every configuration; colors encode families, and marker shapes distinguish settings within a family. The dashed line in the overview connects empirically nondominated measurements and does not imply interpolated or statistically significant performance. Incomplete full-benchmark measurements are omitted and labeled in the relevant legend. Figure and panel titles are omitted; panel identities are specified in this caption and the legends.

### 14. tradeoff_minivla_simplerenv

Placement: Main paper: select representative settings; remaining plots in supplement.

[PNG](png/tradeoff_minivla_simplerenv.png) · [PDF](pdf/tradeoff_minivla_simplerenv.pdf) · [Python](scripts/plot_tradeoff_minivla_simplerenv.py)

Success change in percentage points versus effective time-per-step speedup, relative to the original policy in the same model and environment. Panels in reading order (left to right, top to bottom): all configurations, foveation, action repeat, depth pruning, guarded reuse, and temporal fusion. Family panels use independent axis limits. Black stars and dotted reference lines mark the original policy (1×, 0 pp). Light green marks simultaneous improvement in success and speed. Local legends identify every configuration; colors encode families, and marker shapes distinguish settings within a family. The dashed line in the overview connects empirically nondominated measurements and does not imply interpolated or statistically significant performance. Incomplete full-benchmark measurements are omitted and labeled in the relevant legend. Figure and panel titles are omitted; panel identities are specified in this caption and the legends.

### 15. tradeoff_openvla_simplerenv

Placement: Main paper: select representative settings; remaining plots in supplement.

[PNG](png/tradeoff_openvla_simplerenv.png) · [PDF](pdf/tradeoff_openvla_simplerenv.pdf) · [Python](scripts/plot_tradeoff_openvla_simplerenv.py)

Success change in percentage points versus effective time-per-step speedup, relative to the original policy in the same model and environment. Panels in reading order (left to right, top to bottom): all configurations, foveation, action repeat, depth pruning, guarded reuse, and temporal fusion. Family panels use independent axis limits. Black stars and dotted reference lines mark the original policy (1×, 0 pp). Light green marks simultaneous improvement in success and speed. Local legends identify every configuration; colors encode families, and marker shapes distinguish settings within a family. The dashed line in the overview connects empirically nondominated measurements and does not imply interpolated or statistically significant performance. Incomplete full-benchmark measurements are omitted and labeled in the relevant legend. Figure and panel titles are omitted; panel identities are specified in this caption and the legends.

### 16. tradeoff_openvla_simplerenv_fractal

Placement: Main paper: select representative settings; remaining plots in supplement.

[PNG](png/tradeoff_openvla_simplerenv_fractal.png) · [PDF](pdf/tradeoff_openvla_simplerenv_fractal.pdf) · [Python](scripts/plot_tradeoff_openvla_simplerenv_fractal.py)

Success change in percentage points versus effective time-per-step speedup, relative to the original policy in the same model and environment. Panels in reading order (left to right, top to bottom): all configurations, foveation, action repeat, depth pruning, guarded reuse, and temporal fusion. Family panels use independent axis limits. Black stars and dotted reference lines mark the original policy (1×, 0 pp). Light green marks simultaneous improvement in success and speed. Local legends identify every configuration; colors encode families, and marker shapes distinguish settings within a family. The dashed line in the overview connects empirically nondominated measurements and does not imply interpolated or statistically significant performance. Incomplete full-benchmark measurements are omitted and labeled in the relevant legend. Figure and panel titles are omitted; panel identities are specified in this caption and the legends.

### 17. tradeoff_openvla_libero

Placement: Main paper: select representative settings; remaining plots in supplement.

[PNG](png/tradeoff_openvla_libero.png) · [PDF](pdf/tradeoff_openvla_libero.pdf) · [Python](scripts/plot_tradeoff_openvla_libero.py)

Success change in percentage points versus effective time-per-step speedup, relative to the original policy in the same model and environment. Panels in reading order (left to right, top to bottom): all configurations, foveation, action repeat, depth pruning, guarded reuse, and temporal fusion. Family panels use independent axis limits. Black stars and dotted reference lines mark the original policy (1×, 0 pp). Light green marks simultaneous improvement in success and speed. Local legends identify every configuration; colors encode families, and marker shapes distinguish settings within a family. The dashed line in the overview connects empirically nondominated measurements and does not imply interpolated or statistically significant performance. Incomplete full-benchmark measurements are omitted and labeled in the relevant legend. Figure and panel titles are omitted; panel identities are specified in this caption and the legends.

### 18. tradeoff_smolvla_libero

Placement: Main paper: select representative settings; remaining plots in supplement.

[PNG](png/tradeoff_smolvla_libero.png) · [PDF](pdf/tradeoff_smolvla_libero.pdf) · [Python](scripts/plot_tradeoff_smolvla_libero.py)

Success change in percentage points versus effective time-per-step speedup, relative to the original policy in the same model and environment. Panels in reading order (left to right, top to bottom): all configurations, foveation, action repeat, depth pruning, guarded reuse, and temporal fusion. Family panels use independent axis limits. Black stars and dotted reference lines mark the original policy (1×, 0 pp). Light green marks simultaneous improvement in success and speed. Local legends identify every configuration; colors encode families, and marker shapes distinguish settings within a family. The dashed line in the overview connects empirically nondominated measurements and does not imply interpolated or statistically significant performance. Incomplete full-benchmark measurements are omitted and labeled in the relevant legend. Figure and panel titles are omitted; panel identities are specified in this caption and the legends.

### 19. tradeoff_spatialvla_simplerenv_bridge

Placement: Main paper: select representative settings; remaining plots in supplement.

[PNG](png/tradeoff_spatialvla_simplerenv_bridge.png) · [PDF](pdf/tradeoff_spatialvla_simplerenv_bridge.pdf) · [Python](scripts/plot_tradeoff_spatialvla_simplerenv_bridge.py)

Success change in percentage points versus effective time-per-step speedup, relative to the original policy in the same model and environment. Panels in reading order (left to right, top to bottom): all configurations, foveation, action repeat, depth pruning, guarded reuse, and temporal fusion. Family panels use independent axis limits. Black stars and dotted reference lines mark the original policy (1×, 0 pp). Light green marks simultaneous improvement in success and speed. Local legends identify every configuration; colors encode families, and marker shapes distinguish settings within a family. The dashed line in the overview connects empirically nondominated measurements and does not imply interpolated or statistically significant performance. Incomplete full-benchmark measurements are omitted and labeled in the relevant legend. Figure and panel titles are omitted; panel identities are specified in this caption and the legends.

### 20. tradeoff_spatialvla_simplerenv_fractal

Placement: Main paper: select representative settings; remaining plots in supplement.

[PNG](png/tradeoff_spatialvla_simplerenv_fractal.png) · [PDF](pdf/tradeoff_spatialvla_simplerenv_fractal.pdf) · [Python](scripts/plot_tradeoff_spatialvla_simplerenv_fractal.py)

Success change in percentage points versus effective time-per-step speedup, relative to the original policy in the same model and environment. Panels in reading order (left to right, top to bottom): all configurations, foveation, action repeat, depth pruning, guarded reuse, and temporal fusion. Family panels use independent axis limits. Black stars and dotted reference lines mark the original policy (1×, 0 pp). Light green marks simultaneous improvement in success and speed. Local legends identify every configuration; colors encode families, and marker shapes distinguish settings within a family. The dashed line in the overview connects empirically nondominated measurements and does not imply interpolated or statistically significant performance. Incomplete full-benchmark measurements are omitted and labeled in the relevant legend. Figure and panel titles are omitted; panel identities are specified in this caption and the legends.

### 21. tradeoff_univla_simplerenv_bridge

Placement: Main paper: select representative settings; remaining plots in supplement.

[PNG](png/tradeoff_univla_simplerenv_bridge.png) · [PDF](pdf/tradeoff_univla_simplerenv_bridge.pdf) · [Python](scripts/plot_tradeoff_univla_simplerenv_bridge.py)

Success change in percentage points versus effective time-per-step speedup, relative to the original policy in the same model and environment. Panels in reading order (left to right, top to bottom): all configurations, foveation, action repeat, depth pruning, guarded reuse, and temporal fusion. Family panels use independent axis limits. Black stars and dotted reference lines mark the original policy (1×, 0 pp). Light green marks simultaneous improvement in success and speed. Local legends identify every configuration; colors encode families, and marker shapes distinguish settings within a family. The dashed line in the overview connects empirically nondominated measurements and does not imply interpolated or statistically significant performance. Incomplete full-benchmark measurements are omitted and labeled in the relevant legend. Figure and panel titles are omitted; panel identities are specified in this caption and the legends.

### 22. tradeoff_univla_libero

Placement: Main paper: select representative settings; remaining plots in supplement.

[PNG](png/tradeoff_univla_libero.png) · [PDF](pdf/tradeoff_univla_libero.pdf) · [Python](scripts/plot_tradeoff_univla_libero.py)

Success change in percentage points versus effective time-per-step speedup, relative to the original policy in the same model and environment. Panels in reading order (left to right, top to bottom): all configurations, foveation, action repeat, depth pruning, guarded reuse, and temporal fusion. Family panels use independent axis limits. Black stars and dotted reference lines mark the original policy (1×, 0 pp). Light green marks simultaneous improvement in success and speed. Local legends identify every configuration; colors encode families, and marker shapes distinguish settings within a family. The dashed line in the overview connects empirically nondominated measurements and does not imply interpolated or statistically significant performance. Incomplete full-benchmark measurements are omitted and labeled in the relevant legend. Figure and panel titles are omitted; panel identities are specified in this caption and the legends.

### 23. ablation_foveation

Placement: Main paper or supplement.

[PNG](png/ablation_foveation.png) · [PDF](pdf/ablation_foveation.pdf) · [Python](scripts/plot_ablation_foveation.py)

Foveation configuration sensitivity across all model–environment settings. Left: success change in percentage points; right: effective time-per-step speedup. Lines connect measured configurations only. Categorical reuse/fusion settings are not a continuous numerical scale. Figure and panel titles are omitted; panel identities are specified in this caption and the legends.

### 24. ablation_action_repeat

Placement: Main paper or supplement.

[PNG](png/ablation_action_repeat.png) · [PDF](pdf/ablation_action_repeat.pdf) · [Python](scripts/plot_ablation_action_repeat.py)

Action repeat configuration sensitivity across all model–environment settings. Left: success change in percentage points; right: effective time-per-step speedup. Lines connect measured configurations only. Categorical reuse/fusion settings are not a continuous numerical scale. Figure and panel titles are omitted; panel identities are specified in this caption and the legends.

### 25. ablation_depth_pruning

Placement: Main paper or supplement.

[PNG](png/ablation_depth_pruning.png) · [PDF](pdf/ablation_depth_pruning.pdf) · [Python](scripts/plot_ablation_depth_pruning.py)

Depth pruning configuration sensitivity across all model–environment settings. Left: success change in percentage points; right: effective time-per-step speedup. Lines connect measured configurations only. Categorical reuse/fusion settings are not a continuous numerical scale. Figure and panel titles are omitted; panel identities are specified in this caption and the legends.

### 26. ablation_guarded_reuse

Placement: Main paper or supplement.

[PNG](png/ablation_guarded_reuse.png) · [PDF](pdf/ablation_guarded_reuse.pdf) · [Python](scripts/plot_ablation_guarded_reuse.py)

Guarded reuse configuration sensitivity across all model–environment settings. Left: success change in percentage points; right: effective time-per-step speedup. Lines connect measured configurations only. Categorical reuse/fusion settings are not a continuous numerical scale. Figure and panel titles are omitted; panel identities are specified in this caption and the legends.

### 27. ablation_temporal_fusion

Placement: Main paper or supplement.

[PNG](png/ablation_temporal_fusion.png) · [PDF](pdf/ablation_temporal_fusion.pdf) · [Python](scripts/plot_ablation_temporal_fusion.py)

Temporal fusion configuration sensitivity across all model–environment settings. Left: success change in percentage points; right: effective time-per-step speedup. Lines connect measured configurations only. Categorical reuse/fusion settings are not a continuous numerical scale. Figure and panel titles are omitted; panel identities are specified in this caption and the legends.

### 28. libero_suites_openvla

Placement: Main paper.

[PNG](png/libero_suites_openvla.png) · [PDF](pdf/libero_suites_openvla.pdf) · [Python](scripts/plot_libero_suites_openvla.py)

Success point estimates for every configuration in Spatial, Object, Goal, and Long (libero_10). Each available suite has 100 episodes. A missing point means the suite was not recorded. No across-seed uncertainty is available. Figure and panel titles are omitted; panel identities are specified in this caption and the legends.

### 29. libero_suites_smolvla

Placement: Main paper.

[PNG](png/libero_suites_smolvla.png) · [PDF](pdf/libero_suites_smolvla.pdf) · [Python](scripts/plot_libero_suites_smolvla.py)

Success point estimates for every configuration in Spatial, Object, Goal, and Long (libero_10). Each available suite has 100 episodes. A missing point means the suite was not recorded. No across-seed uncertainty is available. Figure and panel titles are omitted; panel identities are specified in this caption and the legends.

### 30. libero_suites_univla

Placement: Main paper.

[PNG](png/libero_suites_univla.png) · [PDF](pdf/libero_suites_univla.pdf) · [Python](scripts/plot_libero_suites_univla.py)

Success point estimates for every configuration in Spatial, Object, Goal, and Long (libero_10). Each available suite has 100 episodes. A missing point means the suite was not recorded. No across-seed uncertainty is available. Figure and panel titles are omitted; panel identities are specified in this caption and the legends.

### 31. episode_duration_ecdf_bridge

Placement: Supplement.

[PNG](png/episode_duration_ecdf_bridge.png) · [PDF](pdf/episode_duration_ecdf_bridge.pdf) · [Python](scripts/plot_episode_duration_ecdf_bridge.py)

Empirical CDF of all recorded episode durations, including failures, normalized by the original mean within each model and environment. Six prespecified representatives: O, F20, A2, D1, RM, FT (see code key). Duration reflects termination and failure as well as computation; read alongside success. Backbone panels, left to right and then top to bottom: CogACT, CronusVLA, MiniVLA, OpenVLA, SpatialVLA, UniVLA. Figure and panel titles are omitted; panel identities are specified in this caption and the legends.

### 32. episode_duration_ecdf_fractal

Placement: Supplement.

[PNG](png/episode_duration_ecdf_fractal.png) · [PDF](pdf/episode_duration_ecdf_fractal.pdf) · [Python](scripts/plot_episode_duration_ecdf_fractal.py)

Empirical CDF of all recorded episode durations, including failures, normalized by the original mean within each model and environment. Six prespecified representatives: O, F20, A2, D1, RM, FT (see code key). Duration reflects termination and failure as well as computation; read alongside success. Backbone panels, left to right and then top to bottom: CogACT, CronusVLA, OpenVLA, SpatialVLA. Figure and panel titles are omitted; panel identities are specified in this caption and the legends.

### 33. episode_duration_ecdf_libero

Placement: Supplement.

[PNG](png/episode_duration_ecdf_libero.png) · [PDF](pdf/episode_duration_ecdf_libero.pdf) · [Python](scripts/plot_episode_duration_ecdf_libero.py)

Empirical CDF of all recorded episode durations, including failures, normalized by the original mean within each model and environment. Six prespecified representatives: O, F20, A2, D1, RM, FT (see code key). Duration reflects termination and failure as well as computation; read alongside success. Backbone panels, left to right and then top to bottom: OpenVLA, SmolVLA, UniVLA. Figure and panel titles are omitted; panel identities are specified in this caption and the legends.

### 34. cross_setting_consistency

Placement: Main paper.

[PNG](png/cross_setting_consistency.png) · [PDF](pdf/cross_setting_consistency.pdf) · [Python](scripts/plot_cross_setting_consistency.py)

Counts of model–environment settings with lower, identical, or higher observed success than original. Each setting has equal weight. These are descriptive point-estimate comparisons, not significance tests. The incomplete configuration has 12 settings rather than 13. Figure and panel titles are omitted; panel identities are specified in this caption and the legends.

### 35. evaluation_coverage

Placement: Supplement.

[PNG](png/evaluation_coverage.png) · [PDF](pdf/evaluation_coverage.pdf) · [Python](scripts/plot_evaluation_coverage.py)

Total recorded episodes across all 14 configurations by model–environment setting. The hatched segment marks 100 missing episodes: OpenVLA LIBERO conservative fusion has only three of four suites. Figure and panel titles are omitted; panel identities are specified in this caption and the legends.

# Suggested paper figure selection

Start with the following six figures, then adapt the selection to the paper's central claim and page limit. The full catalog retains all settings so individual examples can be checked against the complete results.

| Figure | Purpose |
|---|---|
| `success_change_bridge` | Main result across six WidowX backbones and every intervention setting. |
| `success_change_libero` | Transfer of the intervention effects to LIBERO; explicitly shows the incomplete result. |
| `tradeoff_univla_libero` | Shows why a large time-per-step gain can coexist with a large loss in success. |
| `tradeoff_cogact_base_simplerenv_bridge` | Shows a contrasting setting with improvements from some depth/fusion configurations and limited speed gains. |
| `ablation_action_repeat` | Tests sensitivity to repeat factors across every model–environment setting. |
| `cross_setting_consistency` | Summarizes how often each configuration improves or degrades observed success, without picking a winner separately for each task. |

If Google robot experiments are central, promote `success_change_fractal` to the main paper. If task dependence is central, substitute the appropriate `libero_suites_*` figure. The three `step_speedup_*` plots provide full efficiency results; the other trade-off plots, remaining sweeps, episode-duration CDFs, step-count plots, and coverage bar chart suit the supplement.

The observed pattern motivating this selection: action repeat 2 lowers aggregate success in 10 of 13 settings, leaves one unchanged, and raises it in two. Action repeat 4 lowers success in 12 and leaves one unchanged. Task-aware fusion raises success in five, leaves one unchanged, and lowers it in seven. These are point estimates, not statistical significance statements.

Use the PDFs for LaTeX. All figures are designed at approximately full two-column width (7.16 inches); preserve that width for dense labels. To fit a single column, simplify the figure or regenerate at the intended width instead of shrinking an entire dense figure to half size.

```latex
% Preamble: \usepackage{graphicx}
\begin{figure*}[t]
  \centering
  \includegraphics[width=\textwidth]{paper_figures/pdf/success_change_bridge.pdf}
  \caption{Success-rate change relative to each original policy on
  SimplerEnv WidowX, in percentage points. All 13 intervention
  configurations are shown. Bars show change from the original policy; colors identify intervention families.
  Results are point estimates from the supplied evaluation runs.}
  \label{fig:widowx-success-change}
\end{figure*}
```

Draft captions for all 35 figures are in README.md and figure_manifest.json. Before submission, specify the actual GPU for every setting, verify the scope of the episode timer, identify the exact configurations behind table2's family labels, and add seed replication if making inferential claims. The supplied data do not support confidence intervals across training/evaluation seeds or a claim of statistically significant improvement.

## Trade-off figure design

Each trade-off figure contains an overview and five intervention-family panels with full local legends. Success is shown as percentage-point change from the original policy, and speedup is relative to the same model and environment. Family panels use independent axis limits so small effects remain visible; use the overview for comparisons across families. A black star marks the original policy, the light-green region means higher success and faster time per step, and a dashed line identifies the observed frontier in the overview. These figures are intended for full two-column width.

All figure and panel titles are omitted. Use the draft captions in README.md to identify backbone panels in reading order; axis labels and legends are retained.

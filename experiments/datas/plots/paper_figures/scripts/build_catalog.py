"""Create one independently executable Python entry point per figure."""
from common import *
import re
specs=[]
def add(name,fn,args,caption,priority):
    specs.append(dict(name=name,function=fn,args=args,caption=caption,placement=priority))
for env in ENVS:
    for key,label in [('delta','success_change'),('speedup','step_speedup'),('step_change','episode_steps')]:
        add(f'{label}_{env}','comparison',[env,key],{'delta':'Success-rate change relative to each original policy, in percentage points.','speedup':'Effective time-per-step speedup: original pooled elapsed time per step divided by the configuration value. Includes recorded rollout overhead. Dots show measured speedup on a logarithmic axis; stems connect each measurement to the 1× original-policy reference.','step_change':'Change in mean episode steps relative to original, including successful and failed episodes. Fewer steps alone do not establish better policy performance.'}[key]+' All 13 non-original configurations are shown in aligned backbone panels with shared axes. Color identifies the intervention family. Bars extend from zero for success and episode-step changes. Missing full-benchmark coverage is labeled explicitly.','Main paper' if key!='step_change' else 'Supplement')
for m in models():
    add('tradeoff_'+m,'pareto',[m],'Success change in percentage points versus effective time-per-step speedup, relative to the original policy in the same model and environment. Panels in reading order (left to right, top to bottom): all configurations, foveation, action repeat, depth pruning, guarded reuse, and temporal fusion. Family panels use independent axis limits. Black stars and dotted reference lines mark the original policy (1×, 0 pp). Light green marks simultaneous improvement in success and speed. Local legends identify every configuration; colors encode families, and marker shapes distinguish settings within a family. The dashed line in the overview connects empirically nondominated measurements and does not imply interpolated or statistically significant performance. Incomplete full-benchmark measurements are omitted and labeled in the relevant legend.','Main paper: select representative settings; remaining plots in supplement')
for f in range(1,6):
    add('ablation_'+FAMILIES[f].lower().replace(' ','_'),'ablation',[f],f'{FAMILIES[f]} configuration sensitivity across all model–environment settings. Left: success change in percentage points; right: effective time-per-step speedup. Lines connect measured configurations only. Categorical reuse/fusion settings are not a continuous numerical scale.','Main paper or supplement')
for m in models('libero'):
    add('libero_suites_'+model_name(m).lower(),'suite_success',[m],'Success point estimates for every configuration in Spatial, Object, Goal, and Long (libero_10). Each available suite has 100 episodes. A missing point means the suite was not recorded. No across-seed uncertainty is available.','Main paper')
for env in ENVS:
    add('episode_duration_ecdf_'+env,'duration_ecdf',[env],'Empirical CDF of all recorded episode durations, including failures, normalized by the original mean within each model and environment. Six prespecified representatives: O, F20, A2, D1, RM, FT (see code key). Duration reflects termination and failure as well as computation; read alongside success.','Supplement')
add('cross_setting_consistency','consistency',[],'Counts of model–environment settings with lower, identical, or higher observed success than original. Each setting has equal weight. These are descriptive point-estimate comparisons, not significance tests. The incomplete configuration has 12 settings rather than 13.','Main paper')
add('evaluation_coverage','coverage',[],'Total recorded episodes across all 14 configurations by model–environment setting. The hatched segment marks 100 missing episodes: OpenVLA LIBERO conservative fusion has only three of four suites.','Supplement')
for spec in specs:
    if spec['function'] in ['comparison', 'duration_ecdf']:
        env=spec['args'][0]
        spec['caption'] += ' Backbone panels, left to right and then top to bottom: '+', '.join(model_name(m) for m in models(env))+'.'
    spec['caption'] += ' Figure and panel titles are omitted; panel identities are specified in this caption and the legends.'
    name=spec['name'];fn=spec['function'];args=[name]+spec['args']
    (OUT/'scripts'/('plot_'+name+'.py')).write_text('"""'+spec['caption']+'"""\nfrom common import '+fn+'\n\nif __name__ == "__main__":\n    '+fn+'('+', '.join(repr(a) for a in args)+')\n')
(OUT/'figure_manifest.json').write_text(json.dumps(specs,indent=2))
# Export aggregate estimands and file-level provenance without changing source data.
groups,agg,suites,audit,rows=data()
for filename,records in [('source_metrics.csv',rows),('aggregate_metrics.csv',[{'model':m,'config':c,**v} for (m,c),v in sorted(agg.items())])]:
    with (OUT/'data'/filename).open('w',newline='') as h:
        w=csv.DictWriter(h,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
# Parse primary values in the LaTeX table, retaining its missing values.
txt=(ROOT/'table2.txt').read_text();suite=None;table=[]
for row in txt.split('\\\\'):
    match=re.search(r'\{90\}\{(Long|Goal|Object|Spatial)\}',row)
    if match:suite={'Long':'libero_10','Goal':'libero_goal','Object':'libero_object','Spatial':'libero_spatial'}[match.group(1)]
    cells=row.split('&')
    if len(cells)!=11 or suite is None:continue
    policy=cells[1].strip()
    if policy not in ['Original','Foveation','Action repeat','Depth pruning','Guarded reuse','Temporal fusion']:continue
    for j,m in enumerate(['univla_libero','openvla_libero','smolvla_libero']):
        vals=[]
        for cell in cells[2+3*j:5+3*j]:
            v=re.match(r'\s*(--|\d+\.\d+)',cell);vals.append(float(v.group(1)) if v and v.group(1)!='--' else np.nan)
        family=['Original','Foveation','Action repeat','Depth pruning','Guarded reuse','Temporal fusion'].index(policy)
        candidates=[]
        for i,c in enumerate(CONFIGS):
            d=suites.get((m,c,suite))
            if FAMILY_IDS[i]==family and d and abs(d['success']-vals[0])<.011 and abs(d['steps']-vals[2])<.011:candidates.append(c)
        table.append(dict(model=m,suite=suite,policy=policy,table_success=vals[0],table_latency_ms=vals[1],table_steps=vals[2],matching_configs=';'.join(candidates)))
with (OUT/'data/table2_config_matches.csv').open('w',newline='') as h:
    w=csv.DictWriter(h,fieldnames=list(table[0]));w.writeheader();w.writerows(table)
missing_query=sum(not np.isfinite(v['query_ms']) for v in agg.values())
report={'summary_files':len(rows),'episode_records':sum(len(v) for v in groups.values()),'count_or_success_mismatches':0,'duplicate_task_episode_ids':0,'incomplete_coverage':audit,'aggregate_groups_without_complete_query_timing':missing_query,'table2_rows_parsed':len(table),'table2_rows_without_success_steps_match':[r for r in table if not r['matching_configs']],'font':font_manager.findfont('Times New Roman',fallback_to_default=False)}
(OUT/'data/audit.json').write_text(json.dumps(report,indent=2))
lines=['# Paper figure catalog','',f'{len(specs)} separate figures; each has a Python script, 600-dpi PNG, and vector PDF with embedded Times New Roman.','','## Recommended paper selection','','See [PAPER_SELECTION.md](PAPER_SELECTION.md) for a concrete six-figure starting set and a LaTeX example.','','Use the success-change bar plots and speedup dot plots for the principal benchmarks, 2–3 representative trade-off figures, the relevant configuration sweeps, and the cross-setting consistency plot. Use LIBERO suite plots where task dependence is central. Keep the remaining figures in the supplement; putting every generated figure in the main paper would be repetitive.','','## Reproduce','','From the original plots folder:','','```sh','.venv/bin/python paper_figures/scripts/run_all.py','# Or regenerate one figure:','.venv/bin/python paper_figures/scripts/plot_success_change_libero.py','```','','Fresh environment: `python -m venv .venv` then `.venv/bin/python -m pip install -r paper_figures/requirements.txt`. Times New Roman must be installed; the scripts fail explicitly if unavailable. Source paths are resolved relative to scripts, so invocation does not depend on the working directory.','','## Metrics and scope','','Source of truth: all 699 summaries and their 47,500 episode records in results_corrected. Counts and successes agree for every file. Inputs are untouched. Each model is kept separate by environment. Aggregation pools episodes within a setting; LIBERO has equal-sized suites and the supplied SimplerEnv tasks have equal-sized trial sets. Full comparisons require identical task/episode-ID coverage to original.','','Success is 100 × successes / episodes. Effective time per step is sum(episode_elapsed_ms) / sum(steps_executed), including whatever overhead is in the recorded timer. It is not model-query latency. Speedup is original effective time per step / modified effective time per step. Episode steps and durations include failures. Query latency in exported metrics is weighted by policy-call count and is missing if any episode lacks query timing. No missing query timings are inferred from episode durations.','','All results are descriptive point estimates from the supplied runs. No error bars or significance claims are fabricated; seed-to-seed replication and confirmed pairing protocols are needed for stronger statistical claims. Matching task/episode IDs is a coverage check, not proof of random-seed pairing. Hardware fields are incomplete in table1; cross-model absolute runtime or parameter-size speed claims are not made. Model names and parameter counts in table1 do not by themselves justify parameter-versus-latency regression. No training curves, memory plots, energy plots, or latency violin plots are generated because the necessary measurements are absent.','','## Data limitations and table reconciliation','']
lines += ['- '+x for x in audit]
lines += ['- Table2 uses family-level labels that do not specify every exact configuration. `data/table2_config_matches.csv` records candidates that reproduce success and average steps. Figure labels explicitly state configurations.','- Table2 latency is consistent with elapsed time per environment step in inspected baseline rows, rather than model-query latency. These figures recompute effective step time directly from episode logs. Table2 leaves SmolVLA reuse/fusion latency blank, although episode elapsed times are present; the derived step-time plots use those recorded episode durations and do not claim that query timing was measured.','- Do not interpret empirical Pareto selection on this evaluation set as a validated deployment selection procedure.','', '## Figures and draft captions','']
for i,s in enumerate(specs,1):lines += [f'### {i:02d}. {s["name"]}', '',f'Placement: {s["placement"]}.', '',f'[PNG](png/{s["name"]}.png) · [PDF](pdf/{s["name"]}.pdf) · [Python](scripts/plot_{s["name"]}.py)', '',s['caption'],'']
(OUT/'README.md').write_text('\n'.join(lines))
print(json.dumps(report,indent=2));print('Figures:',len(specs))

"""Reflow Tables I and II into the wide layout (lab meeting 2026-09-15): one row per
backbone and environment, one column pair (Success, Latency) per trick, Avg. Steps
dropped. Cell strings are copied verbatim from simplr.tex and libero.tex, so every
number and colored delta is the mentor's."""
import re
P='/home/user/BiVLA/experiments/paper/tables/'
FAM=['Original','Foveation','Action repeat','Depth pruning','Guarded reuse','Temporal fusion']
def cells(line):
    body=line.split('\\\\')[0]; parts=[c.strip() for c in body.split('&')]
    return parts
# ---- Table I ----
rows={}; model=None; params={}
for line in open(P+'simplr.tex'):
    m=re.search(r'multirow\[c\]\{6\}\{\*\}\{([A-Za-z]+)\}',line)
    if m: model=m.group(1)
    m=re.search(r'multirow\[c\]\{6\}\{\*\}\{([\d.]+B)\}',line)
    if m and model: params[model]=m.group(1)
    m2=re.match(r'\s*(?:&\s*){1,2}(Original|Foveation|Action repeat|Depth pruning|Guarded reuse|Temporal fusion)\s*&(.*)\\\\',line)
    if m2 and model:
        c=[x.strip() for x in m2.group(2).split('&')]
        rows[(model,m2.group(1))]={'WidowX':c[0:3],'Fractal':c[3:6]}
order=['CogACT','OpenVLA','SpatialVLA','CronusVLA','UniVLA','MiniVLA']
out=[]
out.append(r'''\begin{table*}[t]
\centering
\caption{SimplerEnv results on WidowX and Google Robot (Fractal). For each trick the shown setting is the one with the highest success, and values in parentheses are the change from the original method of the same row. Latency is wall-clock per environment step.}
\label{tab:simplerenv-results}
\footnotesize
\setlength{\tabcolsep}{2pt}
\resizebox{\textwidth}{!}{%
\begin{tabular}{llcccccccccccc}
\toprule
\multirow{2}{*}{Model} & \multirow{2}{*}{Env.}
& \multicolumn{2}{c}{Original} & \multicolumn{2}{c}{Foveation} & \multicolumn{2}{c}{Action repeat} & \multicolumn{2}{c}{Depth pruning} & \multicolumn{2}{c}{Guarded reuse} & \multicolumn{2}{c}{Temporal fusion} \\
\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}\cmidrule(lr){9-10}\cmidrule(lr){11-12}\cmidrule(lr){13-14}
& & Succ. $\uparrow$ & Lat. $\downarrow$ & Succ. $\uparrow$ & Lat. $\downarrow$ & Succ. $\uparrow$ & Lat. $\downarrow$ & Succ. $\uparrow$ & Lat. $\downarrow$ & Succ. $\uparrow$ & Lat. $\downarrow$ & Succ. $\uparrow$ & Lat. $\downarrow$ \\
\midrule''')
for mdl in order:
    envs=[e for e in ['WidowX','Fractal'] if rows[(mdl,'Original')][e][0]!='--']
    for k,e in enumerate(envs):
        lab=(r'\multirow{%d}{*}{%s (%s)}'%(len(envs),mdl,params[mdl]) if k==0 else '')
        cs=[]
        for f in FAM:
            v=rows[(mdl,f)][e]; cs+= [v[0],v[1]]
        out.append(f"{lab} & {e} & "+' & '.join(cs)+r' \\')
    out.append(r'\cmidrule(lr){1-14}' if mdl!=order[-1] else '')
out[-1]=r'\bottomrule'
out.append(r'''\end{tabular}%
}
\end{table*}''')
open(P+'simplr_wide.tex','w').write('\n'.join(out)+'\n')
# ---- Table II ----
rows2={}; suite=None; buf=''
for line in open(P+'libero.tex'):
    if line.strip().startswith('%'): continue
    m=re.search(r'multirow.*\{(Long|Goal|Object|Spatial)\}',line)
    if m: suite=m.group(1); buf=''; continue
    if line.strip().startswith(('\\midrule','\\cmidrule','\\bottomrule','\\toprule')): buf=''; continue
    buf+=' '+line.strip()
    if buf.rstrip().endswith('\\\\'):
        m2=re.match(r'\s*&\s*(Original|Foveation|Action repeat|Depth pruning|Guarded reuse|Temporal fusion)\s*&(.*)\\\\',buf.strip())
        if m2 and suite:
            c=[x.strip() for x in m2.group(2).split('&')]
            rows2[(suite,m2.group(1))]={'UniVLA':c[0:3],'OpenVLA':c[3:6],'SmolVLA':c[6:9]}
        buf=''
out=[]
out.append(r'''\begin{table*}[t]
\centering
\caption{LIBERO results across four task suites. For each trick the shown setting is the one with the highest success, and values in parentheses are the change from the original method of the same row. Latency is wall-clock per environment step, and ``--'' marks cells whose latency is not read (Section~IV-A).}
\label{tab:libero-results}
\footnotesize
\setlength{\tabcolsep}{2pt}
\resizebox{\textwidth}{!}{%
\begin{tabular}{llcccccccccccc}
\toprule
\multirow{2}{*}{Suite} & \multirow{2}{*}{Model}
& \multicolumn{2}{c}{Original} & \multicolumn{2}{c}{Foveation} & \multicolumn{2}{c}{Action repeat} & \multicolumn{2}{c}{Depth pruning} & \multicolumn{2}{c}{Guarded reuse} & \multicolumn{2}{c}{Temporal fusion} \\
\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}\cmidrule(lr){9-10}\cmidrule(lr){11-12}\cmidrule(lr){13-14}
& & Succ. $\uparrow$ & Lat. $\downarrow$ & Succ. $\uparrow$ & Lat. $\downarrow$ & Succ. $\uparrow$ & Lat. $\downarrow$ & Succ. $\uparrow$ & Lat. $\downarrow$ & Succ. $\uparrow$ & Lat. $\downarrow$ & Succ. $\uparrow$ & Lat. $\downarrow$ \\
\midrule''')
P2={'UniVLA':'8.5B','OpenVLA':'7B','SmolVLA':'0.45B'}
for si,s in enumerate(['Long','Goal','Object','Spatial']):
    for k,mdl in enumerate(['UniVLA','OpenVLA','SmolVLA']):
        lab=(r'\multirow{3}{*}{%s}'%s if k==0 else '')
        cs=[]
        for f in FAM:
            v=rows2[(s,f)][mdl]; cs+=[v[0], v[1]]
        out.append(f"{lab} & {mdl} ({P2[mdl]}) & "+' & '.join(cs)+r' \\')
    out.append(r'\cmidrule(lr){1-14}' if s!='Spatial' else r'\bottomrule')
out.append(r'''\end{tabular}%
}
\end{table*}''')
open(P+'libero_wide.tex','w').write('\n'.join(out)+'\n')
print('rows I:',sum(1 for l in open(P+'simplr_wide.tex') if l.endswith('\\\\\n')),'rows II:',sum(1 for l in open(P+'libero_wide.tex') if l.endswith('\\\\\n')))

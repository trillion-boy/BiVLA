"""Data loading, documented estimands, and publication style shared by figures."""
from pathlib import Path
import json, csv
from functools import lru_cache
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'paper_figures'
CONFIGS = ['original','fixed_foveation_keep50','fixed_foveation_keep20','action_repeat2','action_repeat4','depth_pruning1','depth_pruning2','depth_pruning4','guarded_reuse_strict','guarded_reuse_moderate','guarded_reuse_aggressive','temporal_fusion_conservative_adaptive','temporal_fusion_motion_entropy','temporal_fusion_task_aware']
LABELS = ['Original','Foveation: 50%','Foveation: 20%','Action repeat: 2','Action repeat: 4','Depth pruning: 1','Depth pruning: 2','Depth pruning: 4','Reuse: strict','Reuse: moderate','Reuse: aggressive','Fusion: conservative','Fusion: motion/entropy','Fusion: task-aware']
SHORT = ['O','F50','F20','A2','A4','D1','D2','D4','RS','RM','RA','FC','FM','FT']
FAMILIES = ['Original','Foveation','Action repeat','Depth pruning','Guarded reuse','Temporal fusion']
COLORS = ['#222222','#0072B2','#D55E00','#009E73','#CC79A7','#8C6510']
MARKERS = ['*','o','s','^','D','P']
FAMILY_IDS = [0,1,1,2,2,3,3,3,4,4,4,5,5,5]
NAMES = {'cogact':'CogACT','cronusvla':'CronusVLA','minivla':'MiniVLA','openvla':'OpenVLA','smolvla':'SmolVLA','spatialvla':'SpatialVLA','univla':'UniVLA'}
ENVS = {'bridge':'SimplerEnv: WidowX','fractal':'SimplerEnv: Google robot','libero':'LIBERO'}

def style():
    for p in Path('/System/Library/Fonts/Supplemental').glob('Times New Roman*.ttf'):
        font_manager.fontManager.addfont(str(p))
    font_manager.findfont('Times New Roman', fallback_to_default=False)
    plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman'],'font.size':8,'axes.titlesize':9,'axes.labelsize':8,'xtick.labelsize':7,'ytick.labelsize':7,'legend.fontsize':7,'pdf.fonttype':42,'ps.fonttype':42,'mathtext.fontset':'stix','axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.6,'lines.linewidth':1.1,'lines.markersize':4,'savefig.dpi':600,'figure.dpi':140,'axes.axisbelow':True})

def model_name(m): return NAMES[m.split('_')[0]]
def env_name(m): return 'libero' if 'libero' in m else ('fractal' if 'fractal' in m else 'bridge')
def finite(v): return isinstance(v,(int,float)) and np.isfinite(v)

def stats(episodes):
    n=len(episodes)
    def total(key):
        vals=[e.get(key) for e in episodes]
        return sum(vals) if all(finite(v) and v>=0 for v in vals) else np.nan
    steps=total('steps_executed'); elapsed=total('episode_elapsed_ms'); calls=total('policy_calls')
    query=[(e.get('query_latency_ms') or {}).get('mean_ms') for e in episodes]
    query_mean=sum(q*e['policy_calls'] for q,e in zip(query,episodes))/calls if calls>0 and all(finite(q) for q in query) else np.nan
    return {'n':n,'successes':sum(bool(e['success']) for e in episodes),'success':100*sum(bool(e['success']) for e in episodes)/n,'steps':steps/n,'episode_s':elapsed/n/1000,'step_ms':elapsed/steps if steps>0 else np.nan,'calls':calls/n,'query_ms':query_mean}

@lru_cache(None)
def data():
    groups={}; suites={}; audit=[]; allrows=[]
    for f in sorted((ROOT/'results_corrected').rglob('summary.json')):
        parts=f.relative_to(ROOT/'results_corrected').parts; m,c=parts[:2]
        d=json.loads(f.read_text()); ep=[json.loads(l) for l in f.with_name('episodes.jsonl').read_text().splitlines() if l.strip()]
        assert len(ep)==d['episodes'] and sum(bool(e['success']) for e in ep)==d['successes'], f
        ids=[(e['task'],e['episode_index']) for e in ep]; assert len(ids)==len(set(ids)),f
        groups.setdefault((m,c),[]).extend(ep)
        suites[(m,c,parts[2] if len(parts)==4 else 'combined')]=stats(ep)
        allrows.append({'source':str(f.relative_to(ROOT)), 'model':m,'config':c,'suite':parts[2] if len(parts)==4 else 'combined',**stats(ep)})
    agg={k:stats(v) for k,v in groups.items()}
    for (m,c),es in groups.items():
        expected={(e['task'],e['episode_index']) for e in groups[(m,'original')]}
        actual={(e['task'],e['episode_index']) for e in es}
        assert len(actual)==len(es),(m,c)
        agg[m,c]['complete']=actual==expected
        if actual!=expected: audit.append(f'{m}/{c}: {len(actual)}/{len(expected)} baseline task/episode IDs; excluded from aggregate comparisons.')
    return groups,agg,suites,audit,allrows

def models(env=None): return sorted({m for m,c in data()[1] if env is None or env_name(m)==env},key=lambda m:(model_name(m), env_name(m)))
def metric(m,c,key):
    a=data()[1].get((m,c),{})
    if not a.get('complete',False): return np.nan
    b=data()[1][m,'original']
    if key=='delta': return a['success']-b['success']
    if key=='speedup': return b['step_ms']/a['step_ms']
    if key=='step_change': return 100*(a['steps']/b['steps']-1)
    if key=='episode_speedup': return b['episode_s']/a['episode_s']
    return a.get(key,np.nan)

def save(fig,name):
    # All publication figures intentionally omit figure and panel titles.
    assert fig._suptitle is None or not fig._suptitle.get_text()
    assert all(not ax.get_title(loc=loc) for ax in fig.axes for loc in ["left", "center", "right"])
    for ext in ['png','pdf']:
        (OUT/ext).mkdir(exist_ok=True)
        fig.savefig(OUT/ext/(name+'.'+ext),bbox_inches='tight',pad_inches=.04,facecolor='white')
    plt.close(fig)

def comparison(name,env,key):
    """Aligned small multiples: geometry encodes magnitude, color identifies family."""
    from matplotlib.ticker import MaxNLocator, FixedLocator, FuncFormatter, NullFormatter
    style(); ms=models(env)
    cols=2 if len(ms)==4 else 3; rows=int(np.ceil(len(ms)/cols))
    fig,axs=plt.subplots(rows,cols,figsize=(7.16,3.6*rows),sharex=True,sharey=True,
                         layout='constrained',squeeze=False)
    y=np.arange(13,dtype=float)
    for boundary in [2,4,7,10]:y[boundary:]+=.6
    values=np.array([[metric(m,c,key) for c in CONFIGS[1:]] for m in ms])
    baseline=1 if key=='speedup' else 0
    lo=min(baseline,float(np.nanmin(values))); hi=max(baseline,float(np.nanmax(values)))
    pad=max((hi-lo)*.09,.025 if key=='speedup' else 2)
    xlim=(lo-pad,hi+pad)
    for ax,m,v in zip(axs.flat,ms,values):
        ax.axvline(baseline,color='#484848',lw=.8,zorder=2)
        for i,value in enumerate(v):
            color=COLORS[FAMILY_IDS[i+1]]
            if not np.isfinite(value):
                ax.text(.98,y[i],'Incomplete coverage',transform=ax.get_yaxis_transform(),
                        ha='right',va='center',fontsize=6.5,color='#666666',style='italic')
                continue
            if key=='speedup':
                ax.plot([baseline,value],[y[i],y[i]],color=color,lw=1.7,zorder=3)
                ax.scatter(value,y[i],s=19,color=color,marker=MARKERS[FAMILY_IDS[i+1]],zorder=4)
            else:
                ax.barh(y[i],value,height=.62,color=color,zorder=3)
                if abs(value)<.0001:ax.plot(0,y[i],marker='|',color=color,markersize=5,zorder=4)
        ax.set_yticks(y,LABELS[1:]);ax.tick_params(axis='y',length=0,pad=5)
        ax.set_ylim(y[-1]+.8,-.8);ax.set_xlim(*xlim)
        ax.spines['left'].set_visible(False);ax.grid(axis='x',color='#e5e5e5',lw=.5)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        if key=='speedup':
            ax.set_xscale('log',base=2)
            ticks=[t for t in [.8,1,1.5,2,3,4] if xlim[0]<=t<=xlim[1]]
            ax.xaxis.set_major_locator(FixedLocator(ticks))
            ax.xaxis.set_major_formatter(FuncFormatter(lambda x,p:f'{x:g}×'))
            ax.xaxis.set_minor_formatter(NullFormatter())
        ax.tick_params(axis='x',labelbottom=True)
    for ax in list(axs.flat)[len(ms):]:ax.set_visible(False)
    titles={'delta':'Success change from original (percentage points)',
            'speedup':'Time-per-step speedup relative to original (log scale)',
            'step_change':'Mean episode step change from original (%)'}
    fig.supxlabel(titles[key],fontsize=9)
    save(fig,name)

def pareto(name,m):
    """Overview plus family-specific detail panels with explicit configuration legends."""
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator, FuncFormatter
    style()
    fig,axs=plt.subplots(2,3,figsize=(7.16,6.5),layout='constrained')
    pts=[(metric(m,c,'speedup'),metric(m,c,'delta'),i) for i,c in enumerate(CONFIGS)]
    pts=[p for p in pts if np.isfinite(p[0]+p[1])]
    symbols=['o','s','^']
    markers={0:'*'}
    for family in range(1,6):
        for j,i in enumerate(k for k,f in enumerate(FAMILY_IDS) if f==family):markers[i]=symbols[j]
    original=Line2D([],[],marker='*',color='#202020',ls='',markersize=8,label='Original policy')
    labels={1:['Keep 50%','Keep 20%'],2:['Repeat 2','Repeat 4'],3:['Prune 1 layer','Prune 2 layers','Prune 4 layers'],
            4:['Strict','Moderate','Aggressive'],5:['Conservative','Motion / entropy','Task-aware']}
    for family,ax in enumerate(axs.flat):
        shown=pts if family==0 else [p for p in pts if p[2]==0 or FAMILY_IDS[p[2]]==family]
        xx=[p[0] for p in shown]; yy=[p[1] for p in shown]
        dx=max(.025,(max(xx)-min(xx))*.18);dy=max(1.1,(max(yy)-min(yy))*.17)
        ax.set_xlim(max(.01,min(xx)-dx),max(xx)+dx)
        ax.set_ylim(min(yy)-dy,max(yy)+dy)
        # Favorable region: faster per step and higher success than the original.
        ax.fill_between([1,ax.get_xlim()[1]],0,ax.get_ylim()[1],color='#EAF3ED',zorder=0)
        ax.axvline(1,color='#777777',ls=':',lw=.8,zorder=1)
        ax.axhline(0,color='#777777',ls=':',lw=.8,zorder=1)
        for x,y,i in shown:
            ax.scatter(x,y,s=70 if i==0 else 29,marker=markers[i],color=COLORS[FAMILY_IDS[i]],
                       edgecolors='white' if i else '#202020',linewidths=.45,zorder=5 if i==0 else 4)
        handles=[]
        if family==0:
            frontier=sorted([(x,y) for x,y,i in pts if not any(xx>=x and yy>=y and (xx>x or yy>y) for xx,yy,j in pts)])
            if len(frontier)>1:ax.plot(*zip(*frontier),ls='--',lw=.8,color='#555555',zorder=2)
            handles=[original]+[Line2D([],[],marker='o',ls='',color=COLORS[f],markersize=4,label=FAMILIES[f]) for f in range(1,6)]+[Line2D([],[],ls='--',lw=.8,color='#555555',label='Observed frontier')]
            ax.text(.97,.96,'Higher + faster',transform=ax.transAxes,ha='right',va='top',fontsize=6,color='#456A50')
        else:
            indices=[i for i,f in enumerate(FAMILY_IDS) if f==family]
            for j,i in enumerate(indices):
                present=any(p[2]==i for p in pts)
                handles.append(Line2D([],[],marker=markers[i],ls='',color=COLORS[family] if present else '#AAAAAA',markersize=4,
                               label=labels[family][j]+(' (incomplete)' if not present else '')))
        ax.legend(handles=handles,loc='upper left',bbox_to_anchor=(0,-.25),ncol=2 if family==0 else 1,
                  frameon=False,fontsize=6.8,handletextpad=.4,columnspacing=.7,borderaxespad=0,labelspacing=.35)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=3,min_n_ticks=3))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4,min_n_ticks=3))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x,p,span=max(xx)-min(xx):f'{x:.2f}' if span<.25 else f'{x:g}'))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x,p:f'{x:+g}' if abs(x)>1e-10 else '0'))
        ax.grid(alpha=.13,lw=.5)
        ax.set_xlabel('Time-per-step speedup (×)',fontsize=7)
        if family%3==0:ax.set_ylabel('Success change (percentage points)',fontsize=7)
    save(fig,name)

def ablation(name,family):
    style(); inds=[i for i,f in enumerate(FAMILY_IDS) if f==family];cs=['original']+[CONFIGS[i] for i in inds]
    labs=['Original']+[LABELS[i].split(': ')[-1] for i in inds]
    fig,axs=plt.subplots(1,2,figsize=(7.16,3.5),layout='constrained')
    palette=plt.get_cmap('tab20')
    for j,m in enumerate(models()):
        for ax,key in zip(axs,['delta','speedup']):
            ax.plot(range(len(cs)),[metric(m,c,key) for c in cs],marker=['o','s','^'][list(ENVS).index(env_name(m))],color=palette(j),label=f'{model_name(m)} / {env_name(m).replace("bridge","WidowX").replace("fractal","Google")}')
            ax.set_xticks(range(len(cs)),labs,rotation=20 if family==5 else 0,ha='right' if family==5 else 'center');ax.grid(alpha=.16)
    axs[0].set_ylabel('Success change (percentage points)');axs[1].set_ylabel('Time-per-step speedup (×)');axs[0].axhline(0,c='#aaaaaa',lw=.6);axs[1].axhline(1,c='#aaaaaa',lw=.6)
    fig.legend(*axs[0].get_legend_handles_labels(),loc='outside lower center',ncol=4,fontsize=6)
    save(fig,name)

def suite_success(name,m):
    style();fig,ax=plt.subplots(figsize=(7.16,4.2),layout='constrained');suites=['libero_spatial','libero_object','libero_goal','libero_10']
    for j,s in enumerate(suites):
        vals=[data()[2].get((m,c,s),{}).get('success',np.nan) for c in CONFIGS]
        ax.plot(vals,np.arange(14)+(j-1.5)*.16,ls='',marker=['o','s','^','D'][j],markersize=4,color=COLORS[j+1],label=['Spatial','Object','Goal','Long'][j])
    ax.set_yticks(range(14),LABELS);ax.invert_yaxis();ax.set(xlim=(-2,103),xlabel='Success (%)');ax.grid(axis='x',alpha=.2);fig.legend(*ax.get_legend_handles_labels(),loc='outside lower center',ncol=4);save(fig,name)

def duration_ecdf(name,env):
    style(); ms=models(env);fig,axs=plt.subplots(1,len(ms),figsize=(7.16,2.7),layout='constrained',squeeze=False)
    selected=['original','fixed_foveation_keep20','action_repeat2','depth_pruning1','guarded_reuse_moderate','temporal_fusion_task_aware']
    for ax,m in zip(axs[0],ms):
        base=metric(m,'original','episode_s')
        for c in selected:
            if not data()[1].get((m,c),{}).get('complete'):continue
            vals=sorted(e['episode_elapsed_ms']/1000/base for e in data()[0][m,c] if finite(e.get('episode_elapsed_ms')))
            i=CONFIGS.index(c);ax.step(vals,np.arange(1,len(vals)+1)/len(vals),where='post',color=COLORS[FAMILY_IDS[i]],label=SHORT[i])
        ax.set_xscale('log');ax.set_ylim(0,1.02);ax.grid(alpha=.15)
        from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter
        ax.xaxis.set_major_locator(FixedLocator([.25,.5,1,2,4]));ax.xaxis.set_major_formatter(FuncFormatter(lambda x,p:f'{x:g}'));ax.xaxis.set_minor_formatter(NullFormatter())
    axs[0,0].set_ylabel('Fraction of episodes');fig.supxlabel('Episode duration / original mean duration (log scale)',fontsize=8)
    fig.legend(*axs[0,0].get_legend_handles_labels(),loc='outside upper center',ncol=6);save(fig,name)

def consistency(name):
    style();fig,ax=plt.subplots(figsize=(7.16,3.8),layout='constrained'); counts=[]
    for c in CONFIGS[1:]:
        v=np.array([metric(m,c,'delta') for m in models()]);v=v[np.isfinite(v)];counts.append([sum(v<-.0001),sum(abs(v)<=.0001),sum(v>.0001)])
    left=np.zeros(13)
    for j,(lab,col) in enumerate(zip(['Lower success','Unchanged success','Higher success'],['#C45A52','#D8D8D8','#377D9C'])):
        v=np.array(counts)[:,j];ax.barh(range(13),v,left=left,color=col,label=lab,height=.72)
        for i,n in enumerate(v):
            if n:ax.text(left[i]+n/2,i,str(n),ha='center',va='center',fontsize=7)
        left+=v
    ax.set_yticks(range(13),LABELS[1:]);ax.invert_yaxis();ax.set_xlabel('Number of model–environment settings (point estimates)');ax.set_xlim(0,13.4);fig.legend(*ax.get_legend_handles_labels(),loc='outside lower center',ncol=3,fontsize=7);save(fig,name)

def coverage(name):
    """Show total recorded coverage and identify the sole missing block directly."""
    from matplotlib.patches import Patch
    style();ms=models();fig,ax=plt.subplots(figsize=(7.16,4.1),layout='constrained')
    envcolors={'bridge':'#0072B2','fractal':'#009E73','libero':'#CC79A7'}
    for j,m in enumerate(ms):
        expected=data()[1][m,'original']['n']*len(CONFIGS)
        recorded=sum(data()[1].get((m,c),{}).get('n',0) for c in CONFIGS)
        ax.barh(j,recorded,height=.65,color=envcolors[env_name(m)])
        if recorded<expected:
            ax.barh(j,expected-recorded,left=recorded,height=.65,facecolor='white',edgecolor='#444444',hatch='////',lw=.7)
            ax.annotate('100 missing: conservative fusion',xy=(expected,j),xytext=(0,12),textcoords='offset points',ha='right',fontsize=6.5,
                        arrowprops=dict(arrowstyle='-',lw=.6,color='#555555'))
    labels=[model_name(m)+' / '+{'bridge':'WidowX','fractal':'Google robot','libero':'LIBERO'}[env_name(m)] for m in ms]
    ax.set_yticks(range(len(ms)),labels);ax.invert_yaxis();ax.tick_params(axis='y',length=0)
    ax.set_xlim(0,6100);ax.spines['left'].set_visible(False);ax.grid(axis='x',alpha=.15)
    ax.set_xlabel('Recorded episodes across all 14 configurations')
    fig.legend(handles=[Patch(facecolor=c,label={'bridge':'WidowX','fractal':'Google robot','libero':'LIBERO'}[e]) for e,c in envcolors.items()],
               loc='outside lower center',ncol=3)
    save(fig,name)

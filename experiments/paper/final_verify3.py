"""Checks for the 2026-09-15 three-question text (IV-B, V, captions)."""
import csv, json, glob, collections
rows=list(csv.DictReader(open('experiments/paper/paired_results_all.csv'))); sig=lambda r: float(r['p_mcnemar'])<0.05
tab=[r for r in rows if r['in_table']=='1']; bb=sorted({r['backbone'] for r in rows}); fails=[]
def chk(n,ok,d=''):
    print(('PASS ' if ok else 'FAIL ')+n,d)
    if not ok: fails.append(n)
R='experiments/datas/plots/results_corrected'
MODEL={'cogact_base_simplerenv_bridge':('CogACT','WidowX'),'cogact_base_simplerenv_fractal':('CogACT','Fractal'),'cronusvla_simplerenv_bridge':('CronusVLA','WidowX'),'cronusvla_simplerenv_fractal':('CronusVLA','Fractal'),'minivla_simplerenv':('MiniVLA','WidowX'),'openvla_simplerenv':('OpenVLA','WidowX'),'openvla_simplerenv_fractal':('OpenVLA','Fractal'),'spatialvla_simplerenv_bridge':('SpatialVLA','WidowX'),'spatialvla_simplerenv_fractal':('SpatialVLA','Fractal'),'univla_simplerenv_bridge':('UniVLA','WidowX'),'openvla_libero':('OpenVLA','LIBERO'),'univla_libero':('UniVLA','LIBERO'),'smolvla_libero':('SmolVLA','LIBERO')}
SUITE={'libero_10':'LIBERO Long','libero_goal':'LIBERO Goal','libero_object':'LIBERO Object','libero_spatial':'LIBERO Spatial'}
D=collections.defaultdict(list)
for m,(b,e) in MODEL.items():
    for f in glob.glob(f'{R}/{m}/*/**/episodes.jsonl', recursive=True):
        cfg=f[len(R)+len(m)+2:].split('/')[0]
        for line in open(f):
            ep=json.loads(line); key=(b,SUITE[ep['task'].split('__')[0]]) if e=='LIBERO' else (b,e); D[(key,cfg)].append(ep)
pairs=sorted({p for p,_ in D}); lat=lambda p,c: sum(e['episode_elapsed_ms'] for e in D[(p,c)])/sum(e['steps_executed'] for e in D[(p,c)]); sp=lambda c:[lat(p,'original')/lat(p,c) for p in pairs]
print('### Fig. 1 (teaser) and RQ-1')
w=[r for r in rows if r['env']=='WidowX']
chk('no WidowX bar far above the original (max gain 9.5)',max(float(r['delta_pts']) for r in w)==9.5)
far=sorted({(r['backbone'],r['family']) for r in w if float(r['delta_pts'])<=-10}); chk('bars far below (<= -10) on WidowX: which tricks',True,far)
chk('110 cells, 22 sig drops, 4 sig gains',(len(tab),sum(sig(r) and float(r['delta_pts'])<0 for r in tab),sum(sig(r) and float(r['delta_pts'])>0 for r in tab))==(110,22,4))
chk('gains on CogACT WidowX, OpenVLA Fractal, UniVLA Goal',{(r['backbone'],r['env']) for r in tab if sig(r) and float(r['delta_pts'])>0}=={('CogACT','WidowX'),('OpenVLA','Fractal'),('UniVLA','LIBERO Goal')})
chk('most drops exceed 10 points (20 of 22)',sum(float(r['delta_pts'])<-10 for r in tab if sig(r) and float(r['delta_pts'])<0)==20)
bd=lambda f: sorted({r['backbone'] for r in tab if r['family']==f and sig(r) and float(r['delta_pts'])<0})
chk('guarded reuse and fusion: no sig drop in any best cell',bd('Guarded reuse')==[] and bd('Temporal fusion')==[])
chk('guarded reuse 0 of 66',sum(r['family']=='Guarded reuse' and sig(r) and float(r['delta_pts'])<0 for r in rows)==0)
md=lambda c: sorted({r['backbone'] for r in rows if r['configuration']==c and sig(r) and float(r['delta_pts'])<0})
chk('mildest setting drops: foveation 3, action repeat 4, depth 4 backbones',(len(md('fixed_foveation_keep50')),len(md('action_repeat2')),len(md('depth_pruning1')))==(3,4,4))
print('### RQ-2')
chk('AR2 1.4 to 1.9x, AR4 up to 3.6x',(round(min(sp('action_repeat2')),1),round(max(sp('action_repeat2')),1),round(max(sp('action_repeat4')),1))==(1.4,1.9,3.6))
chk('AR sig drops 8 of 22 at 2, 20 at 4',[sum(sig(r) and float(r['delta_pts'])<0 for r in rows if r['configuration']==c) for c in ['action_repeat2','action_repeat4']]==[8,20])
chk('depth1 faster beyond floor 14/22, drops 8; depth4 faster 21, drops 14',(sum(v>1.014 for v in sp('depth_pruning1')),sum(sig(r) and float(r['delta_pts'])<0 for r in rows if r['configuration']=='depth_pruning1'),sum(v>1.014 for v in sp('depth_pruning4')),sum(sig(r) and float(r['delta_pts'])<0 for r in rows if r['configuration']=='depth_pruning4'))==(14,8,21,14))
gr=[(p,r['configuration'],lat(p,'original')/lat(p,r['configuration'])) for p in pairs for r in tab if (r['backbone'],r['env'])==p and r['family']=='Guarded reuse' and p[0]!='SmolVLA' and not (p==('CronusVLA','WidowX') and r['configuration']!='guarded_reuse_strict')]
chk('guarded reuse table cells: a tenth at most on OpenVLA LIBERO, under 5 pct elsewhere',max(v for p,c,v in gr)<1.12 and all(v<1.05 for p,c,v in gr if not (p[0]=='OpenVLA' and p[1].startswith('LIBERO'))),round(max(v for p,c,v in gr),3))
chk('foveation slower on every backbone but UniVLA',all(lat(p,'original')/lat(p,c)<0.986 for p in pairs for c in ['fixed_foveation_keep50','fixed_foveation_keep20'] if p[0]!='UniVLA'))
call=lambda p,c: (lambda q: sum(a*b for a,b in q)/sum(b for _,b in q))([(e['query_latency_ms']['mean_ms'],e['policy_calls']) for e in D[(p,c)] if e.get('query_latency_ms') and e['query_latency_ms'].get('mean_ms') is not None])
ta={p:call(p,'temporal_fusion_task_aware')/call(p,'original') for p in pairs if p[0] in ('OpenVLA','MiniVLA','UniVLA')}
chk('task-aware adds about a tenth per call on OpenVLA, MiniVLA, UniVLA (7 to 17 pct)',all(1.07<=v<=1.17 for v in ta.values()),{k:round(v,3) for k,v in ta.items()})
print('### RQ-3')
g=lambda b,e,c: next(r for r in rows if r['backbone']==b and r['env']==e and r['configuration']==c)
chk('depth above original on CogACT, far below on MiniVLA and SpatialVLA (WidowX, budget 2)',float(g('CogACT','WidowX','depth_pruning2')['delta_pts'])>0 and float(g('MiniVLA','WidowX','depth_pruning2')['delta_pts'])<=-17 and float(g('SpatialVLA','WidowX','depth_pruning2')['delta_pts'])<=-17)
chk('foveation keep50 raises UniVLA Goal, lowers SmolVLA Goal (both sig)',float(g('UniVLA','LIBERO Goal','fixed_foveation_keep50')['delta_pts'])>0 and sig(g('UniVLA','LIBERO Goal','fixed_foveation_keep50')) and float(g('SmolVLA','LIBERO Goal','fixed_foveation_keep50')['delta_pts'])<0 and sig(g('SmolVLA','LIBERO Goal','fixed_foveation_keep50')))
chk('AR2 negative on all six WidowX, Fractal none sig',all(float(g(b,'WidowX','action_repeat2')['delta_pts'])<0 for b in ['CogACT','CronusVLA','MiniVLA','OpenVLA','SpatialVLA','UniVLA']) and not any(sig(g(b,'Fractal','action_repeat2')) for b in ['CogACT','CronusVLA','OpenVLA','SpatialVLA']))
print('### V')
ad=lambda f: sorted({r['backbone'] for r in rows if r['family']==f and sig(r) and float(r['delta_pts'])<0})
chk('action repeat lowers success on six of seven',len(ad('Action repeat'))==6); chk('depth1 leaves success intact on three of seven',7-len(md('depth_pruning1'))==3); chk('foveation lowers success on three policies',len(ad('Foveation'))==3)
fs=[r for r in rows if r['family']=='Temporal fusion' and sig(r)]; chk('fusion: sig changes within chance (66 settings, 5 sig, 3 up 2 down)',len(fs)==5 and sum(float(r['delta_pts'])>0 for r in fs)==3)
print('### IV-C')
p=('UniVLA','WidowX'); o={(e['task'],e['episode_index']):e['success'] for e in D[(p,'original')]}
chk('UniVLA WidowX both fusion settings lose 11 net, only ME sig',all(sum(o[(e['task'],e['episode_index'])] for e in D[(p,c)])-sum(e['success'] for e in D[(p,c)])==11 for c in ['temporal_fusion_motion_entropy','temporal_fusion_task_aware']) and sig(g('UniVLA','WidowX','temporal_fusion_motion_entropy')) and not sig(g('UniVLA','WidowX','temporal_fusion_task_aware')))
uv=[(r['env'],sum(e['reuses'] for e in D[(('UniVLA',r['env']),r['configuration'])])) for r in tab if r['backbone']=='UniVLA' and r['family']=='Guarded reuse']; chk('UniVLA reuse cells fired at most five steps',max(v for _,v in uv)==5,uv)
bad=sum(1 for (p,c),v in D.items() for e in v if not e['success'] and e['steps_executed']<((120 if 'eggplant' in e['task'] else 60) if p[1]=='WidowX' else {'Fractal':80,'LIBERO Long':520,'LIBERO Goal':300,'LIBERO Object':280,'LIBERO Spatial':220}[p[1]])); chk('every failed episode reached its cap',bad==0)
print('\nFAILS:',fails)

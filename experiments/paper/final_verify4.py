"""Checks for the 2026-09-15 evening text (IV-A, IV-B, IV-C, V, captions) after the
page-budget cuts and the run-to-run noise correction. Extends final_verify3.py:
the latency floor is now read from all ten zero-fire cells (up to 1.6 percent),
foveation is read over every setting (2 to 17 percent on six backbones, under
4 percent on UniVLA), and run-to-run noise is read from the episodes on which no
reuse gate fired. Run from the repo root."""
import csv, json, glob, collections
rows=list(csv.DictReader(open('experiments/paper/paired_results_all.csv'))); sig=lambda r: float(r['p_mcnemar'])<0.05
tab=[r for r in rows if r['in_table']=='1']; fails=[]
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
print('### IV-A latency floor and run-to-run noise')
zf=[(p,c) for (p,c),eps in D.items() if c.startswith('guarded_reuse') and sum(e['reuses'] for e in eps)==0]
dev=[abs(lat(p,c)/lat(p,'original')-1)*100 for p,c in zf]
chk('ten zero-fire cells, latency within 1.6 percent of the original',len(zf)==10 and max(dev)<=1.6,round(max(dev),2))
noise={}
for p in pairs:
    o={(e['task'],e['episode_index']):e['success'] for e in D[(p,'original')]}; n=d=0
    for c in ['guarded_reuse_strict','guarded_reuse_moderate','guarded_reuse_aggressive']:
        for e in D[(p,c)]:
            if e['reuses']==0: n+=1; d+= o[(e['task'],e['episode_index'])]!=e['success']
    noise[p]=100*d/n
rep=[p for p in pairs if p[0] in ('SpatialVLA','UniVLA') or (p[0]=='OpenVLA' and p[1]!='WidowX')]
non=[p for p in pairs if p[0] in ('CogACT','CronusVLA') or (p[0] in ('MiniVLA','OpenVLA') and p[1]=='WidowX')]
chk('reproducing pairs differ on at most one episode in a hundred',all(noise[p]<=1.0 for p in rep),{f'{p[0]} {p[1]}':round(noise[p],2) for p in rep})
chk('CogACT, CronusVLA, MiniVLA WidowX, OpenVLA WidowX differ on 2 to 17 percent',all(1.5<=noise[p]<=17 for p in non),{f'{p[0]} {p[1]}':round(noise[p],1) for p in non})
chk('SmolVLA reconstructed stack differs on 10 to 30 percent of zero-fire episodes per suite',all(9.5<=noise[p]<=30.5 for p in pairs if p[0]=='SmolVLA'),{p[1]:round(noise[p],1) for p in pairs if p[0]=='SmolVLA'})
chk('V: the four WidowX backbones that do not reproduce',sorted({p[0] for p in pairs if p[1]=='WidowX' and noise[p]>1})==['CogACT','CronusVLA','MiniVLA','OpenVLA'])
print('### RQ-1')
w=[r for r in rows if r['env']=='WidowX']
chk('no WidowX bar more than 10 points above the original',max(float(r['delta_pts']) for r in w)<=10)
chk('bars 10 or more below: action repeat, depth pruning, foveation on OpenVLA and UniVLA',sorted({(r['backbone'],r['family']) for r in w if float(r['delta_pts'])<=-10 and r['family']=='Foveation'})==[('OpenVLA','Foveation'),('UniVLA','Foveation')] and {r['family'] for r in w if float(r['delta_pts'])<=-10}=={'Action repeat','Depth pruning','Foveation'})
chk('110 cells, 22 sig drops, 4 sig gains',(len(tab),sum(sig(r) and float(r['delta_pts'])<0 for r in tab),sum(sig(r) and float(r['delta_pts'])>0 for r in tab))==(110,22,4))
chk('gains on CogACT WidowX, OpenVLA Fractal, UniVLA Goal',{(r['backbone'],r['env']) for r in tab if sig(r) and float(r['delta_pts'])>0}=={('CogACT','WidowX'),('OpenVLA','Fractal'),('UniVLA','LIBERO Goal')})
chk('most drops exceed 10 points (20 of 22)',sum(float(r['delta_pts'])<-10 for r in tab if sig(r) and float(r['delta_pts'])<0)==20)
chk('286 settings: six sig gains, about seven expected',sum(sig(r) and float(r['delta_pts'])>0 for r in rows)==6 and round(286*0.025)==7)
chk('CogACT WidowX: every depth and fusion setting rises',all(float(r['delta_pts'])>0 for r in rows if r['backbone']=='CogACT' and r['env']=='WidowX' and r['family'] in ('Depth pruning','Temporal fusion')))
bd=lambda f: sorted({r['backbone'] for r in tab if r['family']==f and sig(r) and float(r['delta_pts'])<0})
chk('guarded reuse and fusion: no sig drop in any best cell',bd('Guarded reuse')==[] and bd('Temporal fusion')==[])
chk('guarded reuse 0 of 66',sum(r['family']=='Guarded reuse' and sig(r) and float(r['delta_pts'])<0 for r in rows)==0)
ol=[r for r in rows if r['family']=='Guarded reuse' and r['backbone']=='OpenVLA' and r['env'].startswith('LIBERO')]
chk('eleven of twelve OpenVLA LIBERO reuse settings lose points',len(ol)==12 and sum(float(r['delta_pts'])<0 for r in ol)==11)
md=lambda c: sorted({r['backbone'] for r in rows if r['configuration']==c and sig(r) and float(r['delta_pts'])<0})
chk('mildest setting drops: foveation 3, action repeat 4, depth 4 backbones',(len(md('fixed_foveation_keep50')),len(md('action_repeat2')),len(md('depth_pruning1')))==(3,4,4))
print('### RQ-2')
chk('action repeat 1.4 to 3.6x over both settings',(round(min(sp('action_repeat2')),1),round(max(sp('action_repeat4')),1))==(1.4,3.6))
chk('AR sig drops 8 of 22 at 2, 20 at 4',[sum(sig(r) and float(r['delta_pts'])<0 for r in rows if r['configuration']==c) for c in ['action_repeat2','action_repeat4']]==[8,20])
chk('depth1 faster beyond the 1.6 percent floor on 13/22, drops 8; depth4 faster 21, drops 14',(sum(v>1.016 for v in sp('depth_pruning1')),sum(sig(r) and float(r['delta_pts'])<0 for r in rows if r['configuration']=='depth_pruning1'),sum(v>1.016 for v in sp('depth_pruning4')),sum(sig(r) and float(r['delta_pts'])<0 for r in rows if r['configuration']=='depth_pruning4'))==(13,8,21,14))
gr=[(p,r['configuration'],lat(p,'original')/lat(p,r['configuration'])) for p in pairs for r in tab if (r['backbone'],r['env'])==p and r['family']=='Guarded reuse' and p[0]!='SmolVLA' and not (p==('CronusVLA','WidowX') and r['configuration']!='guarded_reuse_strict')]
chk('guarded reuse table cells: a tenth at most on OpenVLA LIBERO, about 5 pct or less elsewhere',max(v for p,c,v in gr)<1.12 and all(v<1.055 for p,c,v in gr if not (p[0]=='OpenVLA' and p[1].startswith('LIBERO'))),round(max(v for p,c,v in gr),3))
fov=[(p,c,(lat(p,c)/lat(p,'original')-1)*100) for p in pairs for c in ['fixed_foveation_keep20','fixed_foveation_keep50']]
six=[d for p,c,d in fov if p[0]!='UniVLA']; uni=[d for p,c,d in fov if p[0]=='UniVLA']
chk('foveation slows six backbones by 2 to 17 percent over every setting',1.5<=min(six) and max(six)<=17.4,(round(min(six),1),round(max(six),1)))
chk('foveation moves UniVLA by under 4 percent',max(abs(d) for d in uni)<4,(round(min(uni),1),round(max(uni),1)))
call=lambda p,c: (lambda q: sum(a*b for a,b in q)/sum(b for _,b in q))([(e['query_latency_ms']['mean_ms'],e['policy_calls']) for e in D[(p,c)] if e.get('query_latency_ms') and e['query_latency_ms'].get('mean_ms') is not None])
ta={p:call(p,'temporal_fusion_task_aware')/call(p,'original') for p in pairs if p[0] in ('OpenVLA','MiniVLA','UniVLA') and p!=('UniVLA','WidowX')}
chk('task-aware adds 7 to 17 percent per call on OpenVLA, MiniVLA, UniVLA (UniVLA WidowX rerun excluded)',all(1.07<=v<=1.17 for v in ta.values()),{k:round(v,3) for k,v in ta.items()})
print('### RQ-3')
g=lambda b,e,c: next(r for r in rows if r['backbone']==b and r['env']==e and r['configuration']==c)
chk('depth above original on CogACT, far below on MiniVLA and SpatialVLA (WidowX, budget 2)',float(g('CogACT','WidowX','depth_pruning2')['delta_pts'])>0 and float(g('MiniVLA','WidowX','depth_pruning2')['delta_pts'])<=-17 and float(g('SpatialVLA','WidowX','depth_pruning2')['delta_pts'])<=-17)
chk('foveation keep50 raises UniVLA Goal, lowers SmolVLA Goal (both sig)',float(g('UniVLA','LIBERO Goal','fixed_foveation_keep50')['delta_pts'])>0 and sig(g('UniVLA','LIBERO Goal','fixed_foveation_keep50')) and float(g('SmolVLA','LIBERO Goal','fixed_foveation_keep50')['delta_pts'])<0 and sig(g('SmolVLA','LIBERO Goal','fixed_foveation_keep50')))
chk('AR2 negative on all six WidowX, sig on four, Fractal none sig',all(float(g(b,'WidowX','action_repeat2')['delta_pts'])<0 for b in ['CogACT','CronusVLA','MiniVLA','OpenVLA','SpatialVLA','UniVLA']) and sum(sig(g(b,'WidowX','action_repeat2')) for b in ['CogACT','CronusVLA','MiniVLA','OpenVLA','SpatialVLA','UniVLA'])==4 and not any(sig(g(b,'Fractal','action_repeat2')) for b in ['CogACT','CronusVLA','OpenVLA','SpatialVLA']))
print('### V')
ad=lambda f: sorted({r['backbone'] for r in rows if r['family']==f and sig(r) and float(r['delta_pts'])<0})
chk('action repeat lowers success on six of seven',len(ad('Action repeat'))==6); chk('depth1 leaves success intact on three of seven',7-len(md('depth_pruning1'))==3); chk('foveation lowers success on three policies',len(ad('Foveation'))==3)
fs=[r for r in rows if r['family']=='Temporal fusion' and sig(r)]; chk('fusion: sig changes within chance (66 settings, 5 sig, 3 up 2 down)',len(fs)==5 and sum(float(r['delta_pts'])>0 for r in fs)==3)
print('### IV-C')
p=('UniVLA','WidowX'); o={(e['task'],e['episode_index']):e['success'] for e in D[(p,'original')]}
chk('UniVLA WidowX both fusion settings lose 11 net, only ME sig',all(sum(o[(e['task'],e['episode_index'])] for e in D[(p,c)])-sum(e['success'] for e in D[(p,c)])==11 for c in ['temporal_fusion_motion_entropy','temporal_fusion_task_aware']) and sig(g('UniVLA','WidowX','temporal_fusion_motion_entropy')) and not sig(g('UniVLA','WidowX','temporal_fusion_task_aware')))
uv=[(r['env'],sum(e['reuses'] for e in D[(('UniVLA',r['env']),r['configuration'])])) for r in tab if r['backbone']=='UniVLA' and r['family']=='Guarded reuse']; chk('UniVLA reuse cells fired at most five steps',max(v for _,v in uv)==5,uv)
bad=sum(1 for (p,c),v in D.items() for e in v if not e['success'] and e['steps_executed']<((120 if 'eggplant' in e['task'] else 60) if p[1]=='WidowX' else {'Fractal':80,'LIBERO Long':520,'LIBERO Goal':300,'LIBERO Object':280,'LIBERO Spatial':220}[p[1]])); chk('every failed episode reached its cap',bad==0)
chk('47,600 episodes, 308 runs',sum(len(v) for v in D.values())==47600 and len(D)==308)
print('### 2026-09-15 evening additions (reviewer pass)')
import math
def p2(w,l):
    n=w+l
    if n==0: return 1.0
    k=min(w,l); return min(1.0,2*sum(math.comb(n,i) for i in range(k+1))/2**n)
def size(n): return sum(math.comb(n,i)/2**n for i in range(n+1) if p2(i,n-i)<0.05) if n else 0.0
s286=sum(size(int(r['win'])+int(r['loss'])) for r in rows)
chk('exact test admits about eight chance passes over 286 settings, about four per direction',7.5<=s286<=8.5,round(s286,2))
chk('exact p recomputed from win and loss matches the CSV',all(abs(p2(int(r['win']),int(r['loss']))-float(r['p_mcnemar']))<0.002 for r in rows))
chk('no table gain exceeds 10 points, a cell at or below -10 in every environment',max(float(r['delta_pts']) for r in tab)<10 and all(any(float(r['delta_pts'])<=-10 for r in tab if r['env']==e) for e in {r['env'] for r in tab}))
allup={(r['backbone'],r['env']) for r in rows if r['family'] in ('Depth pruning','Temporal fusion')}
allup={p for p in allup if all(float(r['delta_pts'])>0 for r in rows if (r['backbone'],r['env'])==p and r['family'] in ('Depth pruning','Temporal fusion'))}
chk('every depth and fusion setting rises on exactly CogACT WidowX and OpenVLA Fractal',allup=={('CogACT','WidowX'),('OpenVLA','Fractal')})
chk('three of the four table gains sit on those two pairs',sum((r['backbone'],r['env']) in allup for r in tab if sig(r) and float(r['delta_pts'])>0)==3)
excl={('SmolVLA','LIBERO Long'),('SmolVLA','LIBERO Goal'),('SmolVLA','LIBERO Object')}
chk('depth4 faster beyond the floor on 18 of the 19 pairs whose latency is read',sum(lat(p,'original')/lat(p,'depth_pruning4')>1.016 for p in pairs if p not in excl)==18 and len([p for p in pairs if p not in excl])==19)
chk('SmolVLA is four of the significant drops at one and at four layers',all(sum(r['backbone']=='SmolVLA' for r in rows if r['configuration']==c and sig(r) and float(r['delta_pts'])<0)==4 for c in ['depth_pruning1','depth_pruning4']))
recon=collections.Counter()
for (p,c),eps in D.items():
    if p[0]=='SmolVLA' and c.startswith('depth'): recon[(p[1],c)]=sum(1 for e in eps if e.get('implementation'))
chk('SmolVLA reconstructed-stack depth episodes: depth2 Long and Goal, depth4 Long, Goal and Object, none at depth1 or Spatial',{k for k,v in recon.items() if v>0}=={('LIBERO Long','depth_pruning2'),('LIBERO Goal','depth_pruning2'),('LIBERO Long','depth_pruning4'),('LIBERO Goal','depth_pruning4'),('LIBERO Object','depth_pruning4')})
fus=[r for r in rows if r['family']=='Temporal fusion' and sig(r) and float(r['delta_pts'])<0]
chk('fusion drops significantly on two settings over all 66 (so [A-1] says table cell)',len(fus)==2)
print('\nFAILS:',fails)

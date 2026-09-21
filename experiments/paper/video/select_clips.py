"""Pick rollout episodes that reflect the paper's aggregate results, encode clips, write clips/manifest.json."""
import imageio_ffmpeg, subprocess, json, os, glob, math
ff=imageio_ffmpeg.get_ffmpeg_exe(); SRC='/home/user/BiVLA/simulation_rollouts'
rows=json.load(open('rollout_index.json')); idx={(r['env'],r['task'],r['k'],r['cfg']):r for r in rows}
# Walls: chosen by agreement with Table I signs (see selection log): one episode per task, three tasks per environment
WALLS=[('widowx','cogact','widowx_put_eggplant_in_basket',3,3.0),('widowx','cogact','widowx_stack_cube',5,2.0),('widowx','cogact','widowx_spoon_on_towel',4,2.0),
       ('fractal','openvla','google_robot_move_near',4,2.0),('fractal','openvla','google_robot_pick_coke_can',4,2.0),('fractal','openvla','google_robot_open_drawer',5,2.0)]
# Pairs on the trick slides use the setting reported in Table I for CogACT WidowX
PAIR_CFG={'fov':'fixed_foveation_keep20','repeat':'action_repeat2','prune':'depth_pruning2','reuse':'guarded_reuse_aggressive','fusion':'temporal_fusion_motion_entropy'}
WANT={'fov':(False,True),'repeat':(True,False),'prune':(False,True),'reuse':(True,True),'fusion':(False,True)}  # (original success, trick success)
def enc(env,model,cfg,task,k,speed,out):
    src=f'{SRC}/{env}/{model}/{cfg}/{task}/rollout_{k}.mp4'; scale='480:360' if env=='widowx' else '224:288'
    d=idx[(env,task,k,cfg)]; cap=str(math.ceil(d['dur']/speed))  # never longer than the whole seconds the slide is held (frame rounding gave 8.1 s clips on an 8 s slide)
    subprocess.run([ff,'-y','-loglevel','error','-i',src,'-vf',f'setpts=PTS/{speed},scale={scale}','-t',cap,'-r','10','-an','-c:v','libx264','-pix_fmt','yuv420p','-profile:v','baseline','-level','3.0','-crf','26','-movflags','+faststart',out],check=True)
    subprocess.run([ff,'-y','-loglevel','error','-i',out,'-frames:v','1',out.replace('.mp4','.png')],check=True)
    d=idx[(env,task,k,cfg)]; return dict(src=src,out=out,success=d['success'],dur=round(d['dur']/speed,1),cfg=cfg)
for f in glob.glob('clips/*'): os.remove(f)
manifest={'walls':[],'pairs':{}}
for env,model,task,k,speed in WALLS:
    tiles=[enc(env,model,cfg,task,k,speed,f'clips/{env}_{task}_k{k}_{cfg}.mp4') for cfg in sorted(os.listdir(f'{SRC}/{env}/{model}'))]
    manifest['walls'].append(dict(env=env,model=model,task=task,k=k,speed=speed,tiles=tiles,maxdur=max(t['dur'] for t in tiles)))
    print('wall',env,task,k,''.join('S' if t['success'] else '.' for t in tiles),'max',max(t['dur'] for t in tiles))
for key,cfg in PAIR_CFG.items():
    wo,wt=WANT[key]; cands=[]
    for (e,task,k,c),r in idx.items():
        if e!='widowx' or c!=cfg: continue
        o=idx[(e,task,k,'original')]
        if o['success']==wo and r['success']==wt: cands.append((max(o['dur'],r['dur']),task,k))
    cands.sort(); dur,task,k=cands[0]
    a=enc('widowx','cogact','original',task,k,2.0,f'clips/pair_{key}_orig.mp4'); b=enc('widowx','cogact',cfg,task,k,2.0,f'clips/pair_{key}_trick.mp4')
    manifest['pairs'][key]=dict(task=task,k=k,cfg=cfg,orig=a,trick=b); print('pair',key,cfg,task,k,'orig',a['success'],'trick',b['success'],'of',len(cands),'candidates')
json.dump(manifest,open('clips/manifest.json','w'),indent=1)
print('MB',round(sum(os.path.getsize(f) for f in glob.glob('clips/*.mp4'))/1e6,1))

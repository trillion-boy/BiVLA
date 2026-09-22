"""Pick rollout episodes that reflect the paper's aggregate results, encode clips, write clips/manifest.json."""
import imageio_ffmpeg, subprocess, json, os, glob, math
ff=imageio_ffmpeg.get_ffmpeg_exe(); SRC='/home/user/BiVLA/simulation_rollouts'
rows=json.load(open('rollout_index.json')); idx={(r['env'],r['task'],r['k'],r['cfg']):r for r in rows}
# Walls: chosen by agreement with Table I signs (see selection log): one episode per task, three tasks per environment
WALLS=[('widowx','cogact','widowx_put_eggplant_in_basket',3,3.0),('widowx','cogact','widowx_stack_cube',5,2.0),('widowx','cogact','widowx_spoon_on_towel',4,2.0),
       ('fractal','openvla','google_robot_move_near',4,2.0),('fractal','openvla','google_robot_pick_coke_can',4,2.0),('fractal','openvla','google_robot_open_drawer',5,2.0)]
# Pairs on the trick slides (mentor's request): the original fails and the trick succeeds; one distinct task per slide,
# chosen from the frames (object clearly placed). env, task, rollout k, trick config, slide seconds, playback speed
PAIRS={'fov':('widowx','widowx_put_eggplant_in_basket',5,'fixed_foveation_keep20',11,2.0),
       'repeat':('fractal','google_robot_pick_coke_can',1,'action_repeat4',11,1.5),
       'prune':('widowx','widowx_stack_cube',3,'depth_pruning1',12,2.0),
       'reuse':('widowx','widowx_carrot_on_plate',3,'guarded_reuse_moderate',11,2.0),
       'fusion':('fractal','google_robot_close_drawer',2,'temporal_fusion_task_aware',12,1.5)}
MODEL={'widowx':'cogact','fractal':'openvla'}
def enc(env,model,cfg,task,k,speed,out,cap=None):
    src=f'{SRC}/{env}/{model}/{cfg}/{task}/rollout_{k}.mp4'; scale='480:360' if env=='widowx' else '224:288'
    d=idx[(env,task,k,cfg)]; cap=str(cap or math.ceil(d['dur']/speed))  # never longer than the whole seconds the slide is held (frame rounding gave 8.1 s clips on an 8 s slide)
    subprocess.run([ff,'-y','-loglevel','error','-i',src,'-vf',f'setpts=PTS/{speed},scale={scale}','-t',cap,'-r','10','-an','-c:v','libx264','-pix_fmt','yuv420p','-profile:v','baseline','-level','3.0','-crf','26','-movflags','+faststart',out],check=True)
    subprocess.run([ff,'-y','-loglevel','error','-i',out,'-frames:v','1',out.replace('.mp4','.png')],check=True)
    d=idx[(env,task,k,cfg)]; return dict(src=src,out=out,success=d['success'],dur=round(d['dur']/speed,1),cfg=cfg)
for f in glob.glob('clips/*'): os.remove(f)
manifest={'walls':[],'pairs':{}}
for env,model,task,k,speed in WALLS:
    tiles=[enc(env,model,cfg,task,k,speed,f'clips/{env}_{task}_k{k}_{cfg}.mp4') for cfg in sorted(os.listdir(f'{SRC}/{env}/{model}'))]
    manifest['walls'].append(dict(env=env,model=model,task=task,k=k,speed=speed,tiles=tiles,maxdur=max(t['dur'] for t in tiles)))
    print('wall',env,task,k,''.join('S' if t['success'] else '.' for t in tiles),'max',max(t['dur'] for t in tiles))
for key,(env,task,k,cfg,secs,speed) in PAIRS.items():
    a=enc(env,MODEL[env],'original',task,k,speed,f'clips/pair_{key}_orig.mp4',cap=secs); b=enc(env,MODEL[env],cfg,task,k,speed,f'clips/pair_{key}_trick.mp4',cap=secs)
    assert (not a['success']) and b['success'], (key,a['success'],b['success'])
    manifest['pairs'][key]=dict(env=env,model=MODEL[env],task=task,k=k,cfg=cfg,speed=speed,orig=a,trick=b); print('pair',key,env,cfg,task,k,'orig',a['success'],'trick',b['success'])
json.dump(manifest,open('clips/manifest.json','w'),indent=1)
print('MB',round(sum(os.path.getsize(f) for f in glob.glob('clips/*.mp4'))/1e6,1))

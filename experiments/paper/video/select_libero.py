"""LIBERO UniVLA walls: one episode per suite, chosen by agreement with Table II (all configurations succeed
except action repeat), encoded at one speed per wall and cut at the slide length. Appends to clips/manifest.json."""
import imageio_ffmpeg, subprocess, json, os, glob, math
ff = imageio_ffmpeg.get_ffmpeg_exe(); SRC = '/home/user/BiVLA/simulation_rollouts/libero/univla'
rows = json.load(open('rollout_index_libero.json')); idx = {(r['task'], r['k'], r['cfg']): r for r in rows}
# (suite task, rollout k, playback speed): highest Table II agreement, then shortest successful clips
WALLS = [('libero_spatial__task_3', 1, 3.0), ('libero_object__task_9', 1, 4.0), ('libero_goal__task_8', 2, 3.0), ('libero_10__task_1', 2, 8.0)]

def enc(cfg, task, k, speed, secs, out):
    src = f'{SRC}/{cfg}/{task}/rollout_{k}.mp4'
    subprocess.run([ff, '-y', '-loglevel', 'error', '-i', src, '-vf', f'setpts=PTS/{speed}', '-t', str(secs), '-r', '10', '-an', '-c:v', 'libx264',
                    '-pix_fmt', 'yuv420p', '-profile:v', 'baseline', '-level', '3.0', '-crf', '27', '-movflags', '+faststart', out], check=True)
    subprocess.run([ff, '-y', '-loglevel', 'error', '-i', out, '-frames:v', '1', out.replace('.mp4', '.png')], check=True)
    d = idx[(task, k, cfg)]
    return dict(src=src, out=out, success=d['success'], dur=round(min(d['dur'] / speed, secs), 1), full=round(d['dur'] / speed, 1), cfg=cfg)

for f in glob.glob('clips/libero_*'): os.remove(f)
M = json.load(open('clips/manifest.json')); M['walls'] = [w for w in M['walls'] if w['env'] != 'libero']
for task, k, speed in WALLS:
    cfgs = sorted(os.listdir(SRC))
    longest_ok = max(idx[(task, k, c)]['dur'] for c in cfgs if idx[(task, k, c)]['success'])
    secs = math.ceil(longest_ok / speed)
    tiles = [enc(c, task, k, speed, secs, f'clips/libero_{task}_k{k}_{c}.mp4') for c in cfgs]
    M['walls'].append(dict(env='libero', model='univla', task=task, k=k, speed=speed, tiles=tiles, maxdur=secs, secs=secs, cut=any(t['full'] > secs for t in tiles)))
    print('wall libero', task, k, f'{speed}x', ''.join('S' if t['success'] else '.' for t in tiles), 'secs', secs, 'cut', sum(t['full'] > secs for t in tiles), 'tiles')
json.dump(M, open('clips/manifest.json', 'w'), indent=1)
print('MB', round(sum(os.path.getsize(f) for f in glob.glob('clips/*.mp4')) / 1e6, 1))

"""Index the two Fractal OpenVLA temporal-fusion presets that arrived later and append them to rollout_index.json."""
import imageio_ffmpeg, subprocess, json, re, io, glob
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
ff = imageio_ffmpeg.get_ffmpeg_exe(); SRC = '/home/user/BiVLA/simulation_rollouts/fractal/openvla'
NEW = ['temporal_fusion_task_aware', 'temporal_fusion_conservative_adaptive']

def one(path):
    cfg, task, fn = path.split('/')[-3:]
    k = int(re.findall(r'\d+', fn)[0])
    pr = subprocess.run([ff, '-i', path], capture_output=True, text=True).stderr
    m = re.search(r'Duration: (\d+):(\d+):([\d.]+)', pr); dur = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))
    png = subprocess.run([ff, '-v', 'error', '-sseof', '-0.3', '-i', path, '-frames:v', '1', '-f', 'image2pipe', '-vcodec', 'png', '-'], capture_output=True).stdout
    im = Image.open(io.BytesIO(png)).convert('RGB'); w, h = im.size
    px = list(im.crop((0, int(h * 0.9), w, h)).getdata())
    g = sum(1 for r, gg, b in px if gg > 120 and r < 110 and b < 110)
    r_ = sum(1 for r, gg, b in px if r > 150 and gg < 90 and b < 90)
    return dict(env='fractal', model='openvla', cfg=cfg, task=task, k=k, dur=round(dur, 1), success=g > r_, g=g, r=r_)

rows = json.load(open('rollout_index.json'))
rows = [r for r in rows if not (r['env'] == 'fractal' and r['cfg'] in NEW)]
files = sorted(f for c in NEW for f in glob.glob(f'{SRC}/{c}/*/rollout_*.mp4'))
with ThreadPoolExecutor(8) as ex: new = list(ex.map(one, files))
amb = [r for r in new if max(r['g'], r['r']) < 300 or min(r['g'], r['r']) > 100]
print(len(new), 'new files; ambiguous badges:', amb)
rows += new
json.dump(rows, open('rollout_index.json', 'w'))
print('index rows', len(rows), 'fractal cfgs', len(set(r['cfg'] for r in rows if r['env'] == 'fractal')))
# check against the experiment records
RC = '/home/user/BiVLA/experiments/datas/plots/results_corrected/openvla_simplerenv_fractal'
mis = 0
for r in new:
    rec = [json.loads(l) for l in open(f"{RC}/{r['cfg']}/{r['task']}/episodes.jsonl")]
    hit = [d for d in rec if d['task'] == r['task'] and d['episode_index'] == r['k'] - 1]
    assert len(hit) == 1, (r, len(hit))
    if hit[0]['success'] != r['success']: mis += 1; print('MISMATCH', r['cfg'], r['task'], r['k'], 'video', r['success'], 'record', hit[0]['success'])
print('new videos vs records: mismatches', mis, 'of', len(new))

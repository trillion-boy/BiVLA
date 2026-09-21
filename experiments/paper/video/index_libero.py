"""Index the LIBERO UniVLA rollouts: duration and SUCCESS/FAILURE badge read from the last frame's bottom band."""
import imageio_ffmpeg, subprocess, json, os, glob, re, io
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
ff = imageio_ffmpeg.get_ffmpeg_exe(); SRC = '/home/user/BiVLA/simulation_rollouts/libero/univla'

def one(path):
    cfg, task, fn = path.split('/')[-3:]
    k = int(re.findall(r'\d+', fn)[0])
    pr = subprocess.run([ff, '-i', path], capture_output=True, text=True).stderr
    m = re.search(r'Duration: (\d+):(\d+):([\d.]+)', pr); dur = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))
    png = subprocess.run([ff, '-v', 'error', '-sseof', '-0.3', '-i', path, '-frames:v', '1', '-f', 'image2pipe', '-vcodec', 'png', '-'], capture_output=True).stdout
    im = Image.open(io.BytesIO(png)).convert('RGB'); w, h = im.size
    band = im.crop((0, int(h * 0.9), w, h)); px = list(band.getdata())
    g = sum(1 for r, gg, b in px if gg > 120 and r < 110 and b < 110)
    r_ = sum(1 for r, gg, b in px if r > 150 and gg < 90 and b < 90)
    suite = task.split('__')[0]
    return dict(env='libero', model='univla', cfg=cfg, suite=suite, task=task, k=k, dur=round(dur, 1), success=g > r_, g=g, r=r_, w=w, h=h)

files = sorted(glob.glob(f'{SRC}/*/*/rollout_*.mp4'))
with ThreadPoolExecutor(8) as ex: rows = list(ex.map(one, files))
amb = [r for r in rows if max(r['g'], r['r']) < 500 or min(r['g'], r['r']) > 200]
print(len(rows), 'files; ambiguous badges:', len(amb)); print(amb[:5])
json.dump(rows, open('rollout_index_libero.json', 'w'))
sizes = set((r['w'], r['h']) for r in rows); print('sizes', sizes)
print('dur min/max', min(r['dur'] for r in rows), max(r['dur'] for r in rows))

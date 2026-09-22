"""Clean the Speechma narration: trim leading/trailing silence, shorten pauses between sentences, normalise loudness.
Writes audio/slideNN.wav plus audio/durations.json. Usage: python3 prep_audio.py <src_dir> <slide_secs_json> [pause=0.35]"""
import sys, os, json, re, subprocess, imageio_ffmpeg
ff = imageio_ffmpeg.get_ffmpeg_exe()
src, secs = sys.argv[1], json.load(open(sys.argv[2])); pause = float(sys.argv[3]) if len(sys.argv) > 3 else 0.35
os.makedirs('audio', exist_ok=True)
# per-slide overrides: (kept pause, tempo)
OVERRIDE = {5: (0.25, 1.0), 13: (0.30, 1.03), 18: (0.30, 1.05)}
def dur(f):
    pr = subprocess.run([ff, '-i', f], capture_output=True, text=True).stderr
    m = re.search(r'Duration: (\d+):(\d+):([\d.]+)', pr); return int(m.group(2)) * 60 + float(m.group(3))
out = {}
for i in range(1, 20):
    p, tempo = OVERRIDE.get(i, (pause, 1.0))
    f = f'{src}/slide{i}.mp3'; o = f'audio/slide{i:02d}.wav'
    af = (f'silenceremove=start_periods=1:start_threshold=-35dB:start_silence=0.15:'
          f'stop_periods=-1:stop_threshold=-35dB:stop_silence={p},'
          f'areverse,silenceremove=start_periods=1:start_threshold=-35dB:start_silence=0.15,areverse,'
          + (f'atempo={tempo},' if tempo != 1.0 else '') + 'loudnorm=I=-16:TP=-1.5:LRA=11,aresample=48000')
    subprocess.run([ff, '-y', '-loglevel', 'error', '-i', f, '-af', af, '-ac', '1', o], check=True)
    d = dur(o); out[i] = round(d, 2)
    print(f'slide {i:2d}: {dur(f):5.1f} s -> {d:5.1f} s  (slide {secs[i-1]} s){"  OVER" if d > secs[i-1] else ""}')
json.dump(out, open('audio/durations.json', 'w'))

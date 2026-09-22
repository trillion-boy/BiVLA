"""Render the accompanying video directly: slide background PNG (1280x720) + the embedded clips overlaid at their slide
positions + the cleaned narration, one segment per slide, concatenated, then encoded to the ICRA limits.
Usage: python3 render_video.py final_secs.json final_layout.json out.mp4"""
import sys, os, json, subprocess, imageio_ffmpeg, re
ff = imageio_ffmpeg.get_ffmpeg_exe()
secs = json.load(open(sys.argv[1])); layout = json.load(open(sys.argv[2])); out = sys.argv[3]
os.makedirs('final_seg', exist_ok=True)
segs = []
for i in range(1, 20):
    T = secs[i - 1]; items = layout[str(i)]
    cmd = [ff, '-y', '-loglevel', 'error', '-loop', '1', '-framerate', '30', '-t', str(T), '-i', f'final_bg/slide{i:02d}.png']
    for it in items: cmd += ['-i', it['clip']]
    cmd += ['-i', f'audio/slide{i:02d}.wav']
    fc = ['[0:v]scale=1280:720:flags=lanczos,format=rgb24[bg]']; last = 'bg'
    for j, it in enumerate(items):
        fc.append(f"[{j+1}:v]scale={it['w']}:{it['h']}:flags=lanczos[c{j}]")
        fc.append(f"[{last}][c{j}]overlay={it['x']}:{it['y']}:eof_action=repeat:shortest=0[v{j}]"); last = f'v{j}'
    fc.append(f'[{last}]fps=30,setsar=1,format=yuv420p[vout]')
    fc.append(f'[{len(items)+1}:a]apad=whole_dur={T},atrim=0:{T},asetpts=N/SR/TB[aout]')
    seg = f'final_seg/seg{i:02d}.mp4'
    cmd += ['-filter_complex', ';'.join(fc), '-map', '[vout]', '-map', '[aout]', '-t', str(T), '-r', '30',
            '-c:v', 'libx264', '-preset', 'fast', '-crf', '18', '-pix_fmt', 'yuv420p', '-c:a', 'aac', '-b:a', '128k', '-ar', '48000', '-ac', '1', seg]
    subprocess.run(cmd, check=True); segs.append(seg)
    print(f'slide {i:2d}: {T} s, {len(items)} clips')
open('final_seg/list.txt', 'w').write(''.join(f"file '{os.path.abspath(s)}'\n" for s in segs))
raw = 'final_seg/ICRA27_raw.mp4'
subprocess.run([ff, '-y', '-loglevel', 'error', '-f', 'concat', '-safe', '0', '-i', 'final_seg/list.txt', '-c', 'copy', '-movflags', '+faststart', raw], check=True)
subprocess.run([ff, '-y', '-loglevel', 'error', '-i', raw, '-vf', 'scale=1280:720', '-r', '30', '-c:v', 'libx264', '-preset', 'slow', '-profile:v', 'high', '-crf', '21', '-maxrate', '800k', '-bufsize', '1600k',
                '-c:a', 'aac', '-b:a', '64k', '-ar', '48000', '-ac', '1', '-movflags', '+faststart', '-map_metadata', '-1', out], check=True)
pr = subprocess.run([ff, '-i', out], capture_output=True, text=True).stderr
m = re.search(r'Duration: (\d+):(\d+):([\d.]+)', pr); d = int(m.group(2)) * 60 + float(m.group(3))
print(f'{out}: {d:.2f} s, {os.path.getsize(out):,} bytes'); print('\n'.join(l.strip() for l in pr.splitlines() if 'Stream' in l))

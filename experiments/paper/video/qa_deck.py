"""QA for a built deck: render every slide to PNG, dump the slide text, check shape bounds and text overlaps,
scan for identifying strings, and write the narration / Speechma files from the speaker notes.
Usage: python3 qa_deck.py deck.pptx outdir [--subs]"""
import sys, os, re, html, zipfile, subprocess, json, glob
pptx, out = sys.argv[1], sys.argv[2]; subs = '--subs' in sys.argv
os.makedirs(out, exist_ok=True)
for f in glob.glob(f'{out}/slide*.png') + glob.glob(f'{out}/*.pdf'): os.remove(f)
subprocess.run(['soffice', '--headless', '--convert-to', 'pdf', '--outdir', out, pptx], capture_output=True)
import pymupdf
pdf = f'{out}/' + os.path.basename(pptx).replace('.pptx', '.pdf'); d = pymupdf.open(pdf)
for i, p in enumerate(d): p.get_pixmap(dpi=110).save(f'{out}/slide{i+1:02d}.png')
z = zipfile.ZipFile(pptx); E = 914400
slides = sorted([n for n in z.namelist() if re.match(r'ppt/slides/slide\d+\.xml$', n)], key=lambda n: int(re.findall(r'\d+', n)[0]))
problems = []; text_dump = []
for n in slides:
    i = int(re.findall(r'\d+', n)[0]); x = z.read(n).decode('utf8')
    boxes = []
    for m in re.finditer(r'<p:(sp|pic)>.*?</p:\1>', x, re.S):
        blk = m.group(0); o = re.search(r'<a:off x="(-?\d+)" y="(-?\d+)"/><a:ext cx="(\d+)" cy="(\d+)"', blk)
        if not o: continue
        X, Y, W, H = [int(v) / E for v in o.groups()]; t = html.unescape(' '.join(re.findall(r'<a:t>(.*?)</a:t>', blk)))
        boxes.append((m.group(1), X, Y, W, H, t))
    text_dump.append(f'--- slide {i}\n' + '\n'.join(b[5] for b in boxes if b[5].strip()))
    band = [b for b in boxes if subs and abs(b[2] - 5.13) < 0.01 and abs(b[4] - 0.495) < 0.01]
    content = [b for b in boxes if b not in band]
    for b in content:
        if b[1] < -0.01 or b[2] < -0.01 or b[1] + b[3] > 10.01 or b[2] + b[4] > 5.626: problems.append((i, 'out of slide', b[5][:40], round(b[2] + b[4], 2)))
        if subs and b[2] + b[4] > 5.13 + 0.005: problems.append((i, 'under subtitle band', b[5][:40] or b[0], round(b[2] + b[4], 2)))
    txt = [b for b in content if b[0] == 'sp' and b[5].strip()]
    for a_i, a in enumerate(txt):
        for b in txt[a_i + 1:]:
            if a[1] < b[1] + b[3] - 0.02 and b[1] < a[1] + a[3] - 0.02 and a[2] < b[2] + b[4] - 0.02 and b[2] < a[2] + a[4] - 0.02:
                # title (y 0.32, h 0.7) and the caption under it (y 0.95) share 0.07 in of box, text never touches: ignore
                if abs(a[2] - 0.32) < 0.01 and abs(b[2] - 0.95) < 0.01: continue
                problems.append((i, 'text overlap', a[5][:30], b[5][:30]))
open(f'{out}/deck_text.txt', 'w').write('\n'.join(text_dump))
alltext = '\n'.join(text_dump).lower()
for bad in ['junseo', 'soumya', 'lin wang', 'github', 'http', 'www.', '@', 'university', 'institute', 'project page', 'anonymous', 'hkust', 'trillion']:
    if bad in alltext: problems.append((0, 'identifying string', bad))
# narration files from notes
notes = sorted([n for n in z.namelist() if re.match(r'ppt/notesSlides/notesSlide\d+\.xml', n)], key=lambda n: int(re.findall(r'\d+', n)[0]))
blocks = []
for n in notes:
    x = z.read(n).decode('utf8').replace('\r', '')
    txt = html.unescape(''.join(re.findall(r'<a:t>(.*?)</a:t>', x, re.S)))
    m = re.search(r'SLIDE (\d+)\s*\|\s*(\d+) s.*?NARRATION:\s*(.*?)\s*VISUAL', txt, re.S)
    blocks.append((int(m.group(1)), int(m.group(2)), ' '.join(m.group(3).split())))
N = len(blocks); total = sum(s for _, s, _ in blocks)
with open(f'{out}/speechma_script.txt', 'w') as f:
    f.write(f'SPEECHMA PASTE FILE (wall deck, {N} slides, {total} s). Block number = slide number. Same English voice for every block.\n\n')
    for i, s, t in blocks: f.write(f'===== Slide {i} of {N} (target {s} s) =====\n{t}\n\n')
with open(f'{out}/narration.md', 'w') as f:
    f.write(f'# Narration, wall deck ({N} slides, {total} s)\n\n| # | s | words | w/s | narration |\n|---|---|---|---|---|\n')
    for i, s, t in blocks: w = len(t.split()); f.write(f'| {i} | {s} | {w} | {w/s:.1f} | {t} |\n')
words = sum(len(t.split()) for _, _, t in blocks)
print(f'{pptx}: {len(d)} pages, {N} notes, {total} s, {words} words; problems: {len(problems)}')
for p in problems: print('  ', p)
json.dump(dict(slides=N, seconds=total, words=words, problems=problems), open(f'{out}/qa.json', 'w'))

"""Build the anonymous project page: static index.html from Markdown sections, CSV tables, figures and clips.
Usage: python3 build.py  (reads content/*.md, data/*.csv; writes site/index.html and copies site/assets)."""
import os, re, csv, shutil, html, json, markdown
HERE = os.path.dirname(os.path.abspath(__file__)); SITE = os.path.join(HERE, 'site'); os.makedirs(os.path.join(SITE, 'assets'), exist_ok=True)
MD = markdown.Markdown(extensions=['tables', 'attr_list', 'toc', 'sane_lists', 'md_in_html'])
def md(path):
    p = os.path.join(HERE, 'content', path)
    if not os.path.exists(p): print('missing', p); return f'<p class="todo">[{path} not written yet]</p>'
    MD.reset(); h = MD.convert(open(p, encoding='utf8').read())
    return h.replace('<table>', '<div class="tablewrap md"><table>').replace('</table>', '</table></div>')
def asset(src, name=None):
    name = name or os.path.basename(src); dst = os.path.join(SITE, 'assets', name)
    if os.path.exists(src): shutil.copyfile(src, dst)
    else: print('missing asset', src)
    return 'assets/' + name

ENV_ORDER = ['WidowX', 'Fractal', 'LIBERO Spatial', 'LIBERO Object', 'LIBERO Goal', 'LIBERO Long']
FAM_ORDER = ['Original', 'Foveation', 'Action repeat', 'Depth pruning', 'Guarded reuse', 'Temporal fusion']
CFG_LABEL = {'original': 'Original', 'fixed_foveation_keep20': 'Foveation, keep 20%', 'fixed_foveation_keep50': 'Foveation, keep 50%',
             'action_repeat2': 'Action repeat, k = 2', 'action_repeat4': 'Action repeat, k = 4',
             'depth_pruning1': 'Depth pruning, 1 layer', 'depth_pruning2': 'Depth pruning, 2 layers', 'depth_pruning4': 'Depth pruning, 4 layers',
             'guarded_reuse_strict': 'Guarded reuse, strict', 'guarded_reuse_moderate': 'Guarded reuse, moderate', 'guarded_reuse_aggressive': 'Guarded reuse, aggressive',
             'temporal_fusion_motion_entropy': 'Temporal fusion, motion-entropy', 'temporal_fusion_task_aware': 'Temporal fusion, task-aware', 'temporal_fusion_conservative_adaptive': 'Temporal fusion, conservative-adaptive'}
CFG_ORDER = list(CFG_LABEL)

def num(v, d=1):
    try: return f'{float(v):.{d}f}'
    except (TypeError, ValueError): return ''
def signed(v, d=1):
    try: x = float(v)
    except (TypeError, ValueError): return ''
    return ('+' if x > 0 else '') + f'{x:.{d}f}' if abs(x) > 1e-9 else '0.0'
def cls_delta(delta, p):
    try: d, pv = float(delta), float(p)
    except (TypeError, ValueError): return ''
    if pv < 0.05: return 'gain' if d > 0 else 'loss'
    return ''

def full_table(rows):
    """One table per environment: rows = configurations, columns = backbones; each cell success (delta, p) / latency / steps."""
    out = []
    envs = [e for e in ENV_ORDER if any(r['env'] == e for r in rows)]
    for env in envs:
        R = [r for r in rows if r['env'] == env]
        backbones = sorted(set(r['backbone'] for r in R), key=lambda b: (['CogACT', 'CronusVLA', 'MiniVLA', 'OpenVLA', 'SpatialVLA', 'UniVLA', 'SmolVLA'].index(b) if b in ['CogACT', 'CronusVLA', 'MiniVLA', 'OpenVLA', 'SpatialVLA', 'UniVLA', 'SmolVLA'] else 99))
        out.append(f'<h4 id="full-{env.lower().replace(" ", "-")}">{html.escape(env)}</h4>')
        out.append('<div class="tablewrap"><table class="full"><thead><tr><th>Configuration</th>' + ''.join(f'<th colspan="3">{b}</th>' for b in backbones) + '</tr>')
        out.append('<tr><th></th>' + ''.join('<th class="sub">Success %</th><th class="sub">ms / step</th><th class="sub">Avg. steps</th>' for b in backbones) + '</tr></thead><tbody>')
        for cfg in CFG_ORDER:
            cells = []
            for b in backbones:
                r = next((x for x in R if x['backbone'] == b and x['configuration'] == cfg), None)
                if r is None: cells.append('<td colspan="3" class="na">n/a</td>'); continue
                if cfg == 'original':
                    cells.append(f'<td class="orig">{num(r["success_pct"])}</td><td>{num(r["latency_ms_per_step"])}</td><td>{num(r["avg_steps"])}</td>')
                else:
                    c = cls_delta(r.get('delta_success_vs_original'), r.get('p_mcnemar'))
                    star = '*' if r.get('in_table') in ('1', 'True', 'true') else ''
                    pv = r.get('p_mcnemar'); pv = ('&lt;0.001' if float(pv) < 0.001 else f'{float(pv):.3f}') if pv not in ('', None) else ''
                    cells.append(f'<td class="{c}">{num(r["success_pct"])}{star}<span class="d">({signed(r.get("delta_success_vs_original"))}, p {pv})</span></td>'
                                 f'<td>{num(r["latency_ms_per_step"])}</td><td>{num(r["avg_steps"])}</td>')
            out.append(f'<tr><td class="cfg">{CFG_LABEL[cfg]}</td>{"".join(cells)}</tr>')
        out.append('</tbody></table></div>')
    return '\n'.join(out)

def per_task_tables(rows):
    out = []
    pairs = []
    for r in rows:
        k = (r['backbone'], r['env'])
        if k not in pairs: pairs.append(k)
    pairs.sort(key=lambda k: (ENV_ORDER.index(k[1]) if k[1] in ENV_ORDER else 99, k[0]))
    for b, env in pairs:
        R = [r for r in rows if r['backbone'] == b and r['env'] == env]
        tasks = []
        for r in R:
            if r['task'] not in tasks: tasks.append(r['task'])
        out.append(f'<details><summary>{b} on {env}: success per task ({len(tasks)} tasks)</summary><div class="tablewrap"><table class="pt"><thead><tr><th>Configuration</th>' + ''.join(f'<th>{html.escape(t)}</th>' for t in tasks) + '</tr></thead><tbody>')
        for cfg in CFG_ORDER:
            cells = []
            for t in tasks:
                r = next((x for x in R if x['configuration'] == cfg and x['task'] == t), None)
                if r is None: cells.append('<td class="na"></td>'); continue
                d = r.get('delta_vs_original', '')
                cells.append(f'<td>{num(r["success_pct"])}' + (f'<span class="d">({signed(d)})</span>' if cfg != 'original' and d != '' else '') + '</td>')
            if any('<td>' in c for c in cells): out.append(f'<tr><td class="cfg">{CFG_LABEL[cfg]}</td>{"".join(cells)}</tr>')
        out.append('</tbody></table></div></details>')
    return '\n'.join(out)

def read_csv(name):
    p = os.path.join(HERE, 'data', name)
    if not os.path.exists(p): print('missing', p); return []
    return list(csv.DictReader(open(p, encoding='utf8')))

CSS = open(os.path.join(HERE, 'style.css'), encoding='utf8').read()
if os.path.isdir(os.path.join(SITE, 'figs')): shutil.rmtree(os.path.join(SITE, 'figs'))
shutil.copytree(os.path.join(HERE, 'figs'), os.path.join(SITE, 'figs'))
full = read_csv('full_settings.csv'); pt = read_csv('per_task.csv')
walls = json.load(open(os.path.join(HERE, 'walls.json'))) if os.path.exists(os.path.join(HERE, 'walls.json')) else []
wall_html = ''.join(f'<figure class="wall"><video src="{asset(w["src"], w["file"])}" controls muted loop playsinline preload="metadata" poster="{asset(w["poster"], w["file"].replace(".mp4", ".jpg"))}"></video><figcaption>{html.escape(w["caption"])}</figcaption></figure>' for w in walls)
figs = json.load(open(os.path.join(HERE, 'figures.json'))) if os.path.exists(os.path.join(HERE, 'figures.json')) else {}
def fig(key):
    f = figs.get(key)
    if not f: return f'<p class="todo">[figure {key} missing]</p>'
    return f'<figure><img src="{asset(f["src"], f["file"])}" alt="{html.escape(f["alt"])}"><figcaption>{f["caption"]}</figcaption></figure>'

page = f'''<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Bag of Tricks for Training-Free VLA Models: project page</title>
<style>{CSS}</style></head>
<body><main>
<header>
<p class="kicker">Anonymous project page for an ICRA 2027 submission</p>
<h1>Bag of Tricks for Training-Free Vision-Language-Action Models:<br>What to See, When to Act, and How Much to Compute?</h1>
{md('intro.md')}
<nav><a href="#video">Video</a> · <a href="#impl">Additional implementation details</a> · <a href="#results">Additional results</a> · <a href="#discussion">Additional discussions</a></nav>
</header>

<section id="video"><h2>Videos</h2>
<video class="main" src="{asset(os.path.join(HERE, 'media', 'accompanying_video.mp4'), 'accompanying_video.mp4')}" controls preload="metadata"></video>
<p class="note">The 3-minute video submitted with the paper: motivation, every configuration of one backbone on one episode for ten episodes, and the results.</p>
<details><summary>Extended video (4.5 minutes) with one slide per trick</summary>
<video class="main" src="{asset(os.path.join(HERE, 'media', 'extended_video.mp4'), 'extended_video.mp4')}" controls preload="metadata"></video>
<p class="note">Same content plus a slide per trick with a paired rollout in which the original policy fails and the trick succeeds. Rollout badges follow the rendered videos.</p>
</details>
</section>

<section id="impl"><h2>1. Additional implementation details</h2>
{md('impl.md')}
</section>

<section id="results"><h2>2. Additional results</h2>
{md('results_intro.md')}
<h3 id="full">2.1 Every setting, every backbone and environment</h3>
{md('full_table_note.md')}
{full_table(full)}
<h3 id="pertask">2.2 Per-task success</h3>
{md('per_task_note.md')}
{per_task_tables(pt)}
<h3 id="figures">2.3 Additional figures</h3>
{md('figures_intro.md')}
{''.join(fig(k) for k in figs)}
<h3 id="stats">2.4 Statistics and reproducibility checks</h3>
{md('stats.md')}
<h3 id="walls">2.5 Qualitative rollouts: every configuration on one episode</h3>
{md('walls_intro.md')}
<div class="walls">{wall_html}</div>
</section>

<section id="discussion"><h2>3. Additional discussions</h2>
{md('discussion.md')}
</section>

<footer><p>Static page, no scripts, no tracking. All numbers come from the per-episode records of the submission.</p></footer>
</main></body></html>'''
open(os.path.join(SITE, 'index.html'), 'w', encoding='utf8').write(page)
print('wrote', os.path.join(SITE, 'index.html'), len(page), 'bytes; full rows', len(full), 'per-task rows', len(pt), 'walls', len(walls), 'figures', len(figs))

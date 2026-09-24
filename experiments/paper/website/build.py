"""Build the anonymous project page: static index.html from Markdown sections, CSV tables, figures and clips.
Usage: python3 build.py  (reads content/*.md, data/*.csv; writes site/index.html and copies site/assets)."""
import os, re, csv, shutil, html, json, markdown
HERE = os.path.dirname(os.path.abspath(__file__)); SITE = os.path.join(HERE, 'site'); os.makedirs(os.path.join(SITE, 'assets'), exist_ok=True)
MD = markdown.Markdown(extensions=['tables', 'attr_list', 'toc', 'sane_lists', 'md_in_html'])
def md(path):
    p = os.path.join(HERE, 'content', path)
    if not os.path.exists(p): print('missing', p); return f'<p class="todo">[{path} not written yet]</p>'
    MD.reset(); h = MD.convert(open(p, encoding='utf8').read())
    h = h.replace('<table>', '<div class="tablewrap md"><table>').replace('</table>', '</table></div>')
    h = re.sub(r'<p>(<img [^>]*>)</p>\s*<p><em>(.*?)</em></p>', r'<figure class="md">\1<figcaption>\2</figcaption></figure>', h, flags=re.S)
    return h
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
        out.append(f'<details{" open" if env == "WidowX" else ""}><summary>{html.escape(env)}: {len(backbones)} backbones, 14 configurations</summary>')
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
        out.append('</tbody></table></div></details>')
    return '\n'.join(out)

TASK_LABEL = {'widowx_carrot_on_plate': 'Carrot on plate', 'widowx_put_eggplant_in_basket': 'Eggplant in basket', 'widowx_spoon_on_towel': 'Spoon on towel', 'widowx_stack_cube': 'Stack cube',
              'google_robot_close_drawer': 'Close drawer', 'google_robot_move_near': 'Move near', 'google_robot_open_drawer': 'Open drawer', 'google_robot_pick_coke_can': 'Pick coke can', 'google_robot_place_apple_in_closed_top_drawer': 'Apple into closed drawer'}
def task_label(t):
    if t in TASK_LABEL: return TASK_LABEL[t]
    m = re.match(r'libero_(?:10|goal|object|spatial)__task_(\d+)$', t)
    return f'Task {m.group(1)}' if m else t
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
        out.append(f'<details><summary>{b} on {env}: success per task ({len(tasks)} tasks)</summary><div class="tablewrap"><table class="pt"><thead><tr><th>Configuration</th>' + ''.join(f'<th>{html.escape(task_label(t))}</th>' for t in tasks) + '</tr></thead><tbody>')
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
os.makedirs(os.path.join(SITE, 'figs'))
_used = set(re.findall(r'figs/([A-Za-z0-9_.-]+\.(?:png|jpg|svg))', ''.join(open(os.path.join(HERE, 'content', f), encoding='utf8').read() for f in os.listdir(os.path.join(HERE, 'content')))))
for _f in sorted(_used): shutil.copyfile(os.path.join(HERE, 'figs', _f), os.path.join(SITE, 'figs', _f))
full = read_csv('full_settings.csv'); pt = read_csv('per_task.csv')
walls = json.load(open(os.path.join(HERE, 'walls.json'))) if os.path.exists(os.path.join(HERE, 'walls.json')) else []
def wall_fig(w): return f'<figure class="wall"><video src="{asset(w["src"], w["file"])}" controls muted loop playsinline preload="metadata" poster="{asset(w["poster"], w["file"].replace(".mp4", ".jpg"))}"></video><figcaption>{html.escape(w["caption"])}</figcaption></figure>'
WALL_FIRST = ['wall_widowx_eggplant.mp4', 'wall_fractal_move_near.mp4', 'wall_libero_goal.mp4', 'wall_libero_long.mp4']
_first = [w for f in WALL_FIRST for w in walls if w['file'] == f]; _rest = [w for w in walls if w not in _first]
wall_html = '<div class="walls">' + ''.join(wall_fig(w) for w in _first) + '</div>' + (f'<details><summary>{len(_rest)} more episodes</summary><div class="walls inner">' + ''.join(wall_fig(w) for w in _rest) + '</div></details>' if _rest else '')
figs = json.load(open(os.path.join(HERE, 'figures.json'))) if os.path.exists(os.path.join(HERE, 'figures.json')) else {}
def fig(key, cls=''):
    f = figs.get(key)
    if not f: return f'<p class="todo">[figure {key} missing]</p>'
    cap = f['caption']
    if f.get('grid'):
        head, _, rest = cap.partition('. ') if '. ' in cap else (cap.rstrip('.'), '', '')
        head = head.split(':')[0]
        rest = (cap[len(head) + 1:].lstrip(': .') if cap.startswith(head) else cap)
        cap = f'<strong>{html.escape(head)}.</strong> ' + (rest[:1].upper() + rest[1:] if rest else '')
    alt = html.escape(f['alt'] if not f.get('grid') else f['caption'].split(':')[0].rstrip('.'))
    return f'<figure class="{cls}"><img src="{asset(f["src"], f["file"])}" alt="{alt}"><figcaption>{cap}</figcaption></figure>'
def tradeoff_block():
    keys = [k for k in figs if figs[k].get('grid')]
    return (f'<details><summary>Trade-off plots for the {len(keys)} backbone and environment pairs that Fig. 4 does not show</summary>'
            + ''.join(fig(k, 'tradeoff') for k in keys) + '</details>')

page = f'''<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Bag of Tricks for Training-Free VLA Models: project page</title>
<style>{CSS}</style></head>
<body><main>
<header>
<p class="kicker">Anonymous project page for an ICRA 2027 submission</p>
<h1>Bag of Tricks for Training-Free Vision-Language-Action Models<span class="sub">What to See, When to Act, and How Much to Compute?</span></h1>
{md('intro.md')}
<nav><a href="#video">Videos</a><a href="#impl">1. Implementation details</a><a href="#results">2. Results</a><a href="#discussion">3. Discussions</a></nav>
</header>

<section id="video"><h2>Videos</h2>
<div class="videobox"><video class="main" src="{asset(os.path.join(HERE, 'media', 'accompanying_video.mp4'), 'accompanying_video.mp4')}" controls preload="metadata"></video></div>
<p class="note">The 3-minute video submitted with the paper.</p>
<details><summary>Extended video with one slide per trick</summary>
<video class="main" src="{asset(os.path.join(HERE, 'media', 'extended_video.mp4'), 'extended_video.mp4')}" controls preload="metadata"></video>
<p class="note">Same content plus one slide per trick with a paired rollout in which the original policy fails and the trick succeeds.</p>
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
<h3 id="figures">2.3 Figures for Fractal and LIBERO</h3>
{md('figures_intro.md')}
{''.join(fig(k) for k in figs if not figs[k].get('grid'))}
<h3 id="stats">2.4 Statistics and reproducibility checks</h3>
{md('stats.md')}
<h3 id="walls">2.5 Qualitative rollouts: every configuration on one episode</h3>
{md('walls_intro.md')}
{wall_html}
</section>

<section id="discussion"><h2>3. Additional discussions</h2>
{md('discussion.md')}
</section>

<footer><p>Static page, no scripts, no tracking. All numbers come from the per-episode records of the submission.</p></footer>
</main></body></html>'''
open(os.path.join(SITE, 'index.html'), 'w', encoding='utf8').write(page)
print('wrote', os.path.join(SITE, 'index.html'), len(page), 'bytes; full rows', len(full), 'per-task rows', len(pt), 'walls', len(walls), 'figures', len(figs))

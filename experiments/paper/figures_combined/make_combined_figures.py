"""Candidate compact layouts for Soumya's item 3 ("plots are too big, combine").
Data, colors, markers and estimands are the mentor's (common.py) unchanged;
only the subplot grid and figure size differ. Output goes to this scratchpad
folder, never to the catalog paper_figures/pdf."""
import sys, json
from pathlib import Path
sys.path.insert(0, '/home/user/BiVLA/experiments/datas/plots/paper_figures/scripts')
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import MaxNLocator
import common

HERE = Path(__file__).resolve().parent
common.OUT = HERE  # keep the catalog untouched


def style_fallback():
    for p in font_manager.findSystemFonts():
        if 'LiberationSerif' in p:
            font_manager.fontManager.addfont(p)
    plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Liberation Serif'], 'font.size': 8,
                         'axes.titlesize': 9, 'axes.labelsize': 8, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
                         'legend.fontsize': 7, 'pdf.fonttype': 42, 'ps.fonttype': 42, 'mathtext.fontset': 'stix',
                         'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': .6,
                         'lines.linewidth': 1.1, 'lines.markersize': 4, 'savefig.dpi': 300, 'figure.dpi': 140,
                         'axes.axisbelow': True})


common.style = style_fallback
groups, agg, suites, audit, allrows = common.data()

# Pool the OpenVLA LIBERO conservative cell with its Long summary, as for Fig. 2.
m, c = 'openvla_libero', 'temporal_fusion_conservative_adaptive'
long = json.loads((common.ROOT.parent / 'OpenVLA_Libero_Long.json').read_text())
assert long['config_name'] == c and long['task_suite_name'] == 'libero_10'
parts = [suites[(m, c, s)] for s in ('libero_goal', 'libero_object', 'libero_spatial')]
succ = sum(p['successes'] for p in parts) + long['successes']
n = sum(p['n'] for p in parts) + long['episodes']
agg[(m, c)].update({'successes': succ, 'n': n, 'success': 100 * succ / n, 'complete': True})


def comparison_row(name, env, key='delta', height=2.3):
    """comparison() with one row of panels instead of a 2x3 grid."""
    common.style(); ms = common.models(env)
    fig, axs = plt.subplots(1, len(ms), figsize=(7.16, height), sharex=True, sharey=True,
                            layout='constrained', squeeze=False)
    y = np.arange(13, dtype=float)
    for boundary in [2, 4, 7, 10]: y[boundary:] += .6
    values = np.array([[common.metric(mm, cc, key) for cc in common.CONFIGS[1:]] for mm in ms])
    baseline = 0
    lo = min(baseline, float(np.nanmin(values))); hi = max(baseline, float(np.nanmax(values)))
    pad = max((hi - lo) * .09, 2); xlim = (lo - pad, hi + pad)
    for ax, mm, v in zip(axs.flat, ms, values):
        ax.axvline(baseline, color='#484848', lw=.8, zorder=2)
        for i, value in enumerate(v):
            color = common.COLORS[common.FAMILY_IDS[i + 1]]
            ax.barh(y[i], value, height=.62, color=color, zorder=3)
            if abs(value) < .0001: ax.plot(0, y[i], marker='|', color=color, markersize=5, zorder=4)
        ax.set_yticks(y, common.LABELS[1:]); ax.tick_params(axis='y', length=0, pad=4)
        ax.set_ylim(y[-1] + .8, -.8); ax.set_xlim(*xlim)
        ax.spines['left'].set_visible(False); ax.grid(axis='x', color='#e5e5e5', lw=.5)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=3))
        ax.tick_params(axis='x', labelbottom=True)
    fig.supxlabel('Success change from dense (percentage points)', fontsize=8)
    common.save(fig, name)


def ablation_row(name, families=(3, 2), height=2.1):
    """ablation() for two families side by side: four panels in one row, one legend."""
    common.style(); palette = plt.get_cmap('tab20')
    fig, axs = plt.subplots(1, 4, figsize=(7.16, height), layout='constrained')
    for k, family in enumerate(families):
        inds = [i for i, f in enumerate(common.FAMILY_IDS) if f == family]
        cs = ['original'] + [common.CONFIGS[i] for i in inds]
        labs = ['Original'] + [common.LABELS[i].split(': ')[-1] for i in inds]
        for j, mm in enumerate(common.models()):
            for ax, key in zip(axs[2 * k:2 * k + 2], ['delta', 'speedup']):
                ax.plot(range(len(cs)), [common.metric(mm, cc, key) for cc in cs],
                        marker=['o', 's', '^'][list(common.ENVS).index(common.env_name(mm))], color=palette(j),
                        label=f'{common.model_name(mm)} / {common.env_name(mm).replace("bridge", "WidowX").replace("fractal", "Fractal").replace("libero", "LIBERO")}')
                ax.set_xticks(range(len(cs)), labs); ax.grid(alpha=.16)
        axs[2 * k].set_ylabel('Success change (points)'); axs[2 * k + 1].set_ylabel('Time-per-step speedup (×)')
        axs[2 * k].axhline(0, c='#aaaaaa', lw=.6); axs[2 * k + 1].axhline(1, c='#aaaaaa', lw=.6)
    fig.legend(*axs[0].get_legend_handles_labels(), loc='outside lower center', ncol=5, fontsize=6)
    common.save(fig, name)


def consistency_col(name, size=(3.5, 2.5)):
    """consistency() at single-column width."""
    common.style(); fig, ax = plt.subplots(figsize=size, layout='constrained'); counts = []
    for cc in common.CONFIGS[1:]:
        v = np.array([common.metric(mm, cc, 'delta') for mm in common.models()]); v = v[np.isfinite(v)]
        counts.append([sum(v < -.0001), sum(abs(v) <= .0001), sum(v > .0001)])
    assert all(sum(r) == 13 for r in counts), counts
    left = np.zeros(13)
    for j, (lab, col) in enumerate(zip(['Lower success', 'Unchanged success', 'Higher success'], ['#C45A52', '#D8D8D8', '#377D9C'])):
        v = np.array(counts)[:, j]; ax.barh(range(13), v, left=left, color=col, label=lab, height=.72)
        for i, k in enumerate(v):
            if k: ax.text(left[i] + k / 2, i, str(k), ha='center', va='center', fontsize=6.5)
        left += v
    ax.set_yticks(range(13), common.LABELS[1:]); ax.invert_yaxis()
    ax.set_xlabel('Number of backbone and environment combinations'); ax.set_xlim(0, 13.4)
    fig.legend(*ax.get_legend_handles_labels(), loc='outside lower center', ncol=3, fontsize=6.5)
    common.save(fig, name)


comparison_row('cand_success_change_bridge_row', 'bridge')
ablation_row('cand_ablation_depth_repeat_row')
consistency_col('cand_cross_setting_consistency_col')
print('done')

"""Single-column versions of the two Section IV row figures (2026-09-15, professor:
figures on one side of the page, not across it). Data, colors and markers are the
mentor's common.py; only layout changes."""
import sys
from pathlib import Path
sys.path.insert(0, '/home/user/BiVLA/experiments/datas/plots/paper_figures/scripts')
import numpy as np, matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, FuncFormatter
import common
HERE = Path(__file__).resolve().parent
for p in font_manager.findSystemFonts():
    if 'LiberationSerif' in p: font_manager.fontManager.addfont(p)
plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Liberation Serif'], 'font.size': 7,
    'axes.labelsize': 7, 'xtick.labelsize': 6, 'ytick.labelsize': 6, 'legend.fontsize': 5.8,
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'mathtext.fontset': 'stix', 'axes.spines.top': False,
    'axes.spines.right': False, 'axes.linewidth': .5, 'lines.linewidth': .9, 'lines.markersize': 3, 'axes.axisbelow': True})

def save(fig, name):
    for ext in ['pdf', 'png']:
        (HERE / ext).mkdir(exist_ok=True)
        fig.savefig(HERE / ext / f'{name}.{ext}', bbox_inches='tight', pad_inches=.03, dpi=300, facecolor='white')

def tradeoff_col(name, env='bridge', size=(3.5, 3.5)):
    ms = common.models(env)
    fig, axs = plt.subplots(2, 3, figsize=size, layout='constrained')
    symbols = ['o', 's', '^']; markers = {0: '*'}
    for family in range(1, 6):
        for j, i in enumerate(k for k, f in enumerate(common.FAMILY_IDS) if f == family): markers[i] = symbols[j]
    for ax, mm in zip(axs.flat, ms):
        pts = [(common.metric(mm, c, 'speedup'), common.metric(mm, c, 'delta'), i) for i, c in enumerate(common.CONFIGS)]
        pts = [p for p in pts if np.isfinite(p[0] + p[1])]
        xx = [p[0] for p in pts]; yy = [p[1] for p in pts]
        dx = max(.06, (max(xx) - min(xx)) * .15); dy = max(3, (max(yy) - min(yy)) * .12)
        ax.set_xlim(min(xx) - dx, max(xx) + dx); ax.set_ylim(min(yy) - dy, max(yy) + dy)
        ax.fill_between([1, ax.get_xlim()[1]], 0, ax.get_ylim()[1], color='#EAF3ED', zorder=0)
        ax.axvline(1, color='#777777', ls=':', lw=.6, zorder=1); ax.axhline(0, color='#777777', ls=':', lw=.6, zorder=1)
        for x, y, i in pts:
            ax.scatter(x, y, s=42 if i == 0 else 14, marker=markers[i], color=common.COLORS[common.FAMILY_IDS[i]],
                       edgecolors='white' if i else '#202020', linewidths=.35, zorder=5 if i == 0 else 4)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=3, min_n_ticks=2)); ax.yaxis.set_major_locator(MaxNLocator(nbins=4, min_n_ticks=3))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{x:g}'))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{x:+g}' if abs(x) > 1e-10 else '0'))
        ax.tick_params(length=2, pad=1.5); ax.grid(alpha=.13, lw=.4)
    for ax in axs[1]: ax.set_xlabel('Speedup (×)')
    for ax in axs[:, 0]: ax.set_ylabel('Success change (points)')
    labels = {1: ['Keep 50%', 'Keep 20%'], 2: ['Repeat 2', 'Repeat 4'], 3: ['Prune 1 layer', 'Prune 2 layers', 'Prune 4 layers'],
              4: ['Reuse strict', 'Reuse moderate', 'Reuse aggressive'], 5: ['Fusion conservative', 'Fusion motion/entropy', 'Fusion task-aware']}
    def h(i):
        f = common.FAMILY_IDS[i]; j = sum(1 for k in range(i) if common.FAMILY_IDS[k] == f)
        return Line2D([], [], marker=markers[i], ls='', color=common.COLORS[f], markersize=3.5, label=labels[f][j])
    original = Line2D([], [], marker='*', color='#202020', ls='', markersize=6, label='Original')
    # rows of three, read left to right: original+foveation / repeat+prune1 / prune2,3+... keep each trick together where possible
    order = [original, h(1), h(2), h(3), h(4), Line2D([], [], ls='', marker='', label=' '), h(5), h(6), h(7), h(8), h(9), h(10), h(11), h(12), h(13)]
    ncol = 3; nrow = 5
    interleaved = [order[r + c * nrow] for c in range(ncol) for r in range(nrow)]  # matplotlib fills columns first
    interleaved = [order[c + r * ncol] for c in range(ncol) for r in range(nrow)]
    fig.legend(handles=interleaved, loc='outside lower center', ncol=ncol, frameon=False, handletextpad=.3, columnspacing=1.0, labelspacing=.25)
    save(fig, name)

def ablation_col(name, families=(3, 2), size=(3.5, 3.9)):
    palette = plt.get_cmap('tab20')
    fig, axs = plt.subplots(2, 2, figsize=size, layout='constrained')
    for k, family in enumerate(families):
        inds = [i for i, f in enumerate(common.FAMILY_IDS) if f == family]
        cs = ['original'] + [common.CONFIGS[i] for i in inds]
        labs = ['Orig.'] + [common.LABELS[i].split(': ')[-1] for i in inds]
        for j, mm in enumerate(common.models()):
            for ax, key in zip(axs[k], ['delta', 'speedup']):
                ax.plot(range(len(cs)), [common.metric(mm, cc, key) for cc in cs],
                        marker=['o', 's', '^'][list(common.ENVS).index(common.env_name(mm))], color=palette(j),
                        label=f'{common.model_name(mm)} / {common.env_name(mm).replace("bridge", "WidowX").replace("fractal", "Fractal").replace("libero", "LIBERO")}')
                ax.set_xticks(range(len(cs)), labs); ax.grid(alpha=.16, lw=.4); ax.tick_params(length=2, pad=1.5)
        axs[k][0].set_ylabel('Success change (points)'); axs[k][1].set_ylabel('Speedup (×)')
        axs[k][0].axhline(0, c='#aaaaaa', lw=.5); axs[k][1].axhline(1, c='#aaaaaa', lw=.5)
    axs[0][0].set_xlabel('Removed layers'); axs[0][1].set_xlabel('Removed layers')
    axs[1][0].set_xlabel('Repeat factor'); axs[1][1].set_xlabel('Repeat factor')
    fig.legend(*axs[0][0].get_legend_handles_labels(), loc='outside lower center', ncol=3, frameon=False, handletextpad=.3, columnspacing=.8, labelspacing=.25)
    save(fig, name)

tradeoff_col('tradeoff_bridge_col')
ablation_col('ablation_depth_repeat_col')
print('done')

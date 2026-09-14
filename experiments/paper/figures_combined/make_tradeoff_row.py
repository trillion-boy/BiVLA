"""Fig. 2 replacement (lab meeting 2026-09-15): six small trade-off panels, one per
WidowX backbone, all 14 settings including the original as a star. Data, colors,
markers and estimands are the mentor's common.py, unchanged; only the layout differs."""
import sys
from pathlib import Path
sys.path.insert(0, '/home/user/BiVLA/experiments/datas/plots/paper_figures/scripts')
import numpy as np, matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, FuncFormatter
import common
HERE = Path(__file__).resolve().parent
common.OUT = HERE

def style_fallback():
    for p in font_manager.findSystemFonts():
        if 'LiberationSerif' in p: font_manager.fontManager.addfont(p)
    plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Liberation Serif'], 'font.size': 8,
        'axes.labelsize': 8, 'xtick.labelsize': 6.5, 'ytick.labelsize': 6.5, 'legend.fontsize': 6.5,
        'pdf.fonttype': 42, 'ps.fonttype': 42, 'mathtext.fontset': 'stix', 'axes.spines.top': False,
        'axes.spines.right': False, 'axes.linewidth': .6, 'lines.markersize': 4, 'savefig.dpi': 300, 'axes.axisbelow': True})
common.style = style_fallback

def tradeoff_row(name, env='bridge', height=2.45):
    common.style(); ms = common.models(env)
    fig, axs = plt.subplots(1, len(ms), figsize=(7.16, height), layout='constrained')
    symbols = ['o', 's', '^']; markers = {0: '*'}
    for family in range(1, 6):
        for j, i in enumerate(k for k, f in enumerate(common.FAMILY_IDS) if f == family): markers[i] = symbols[j]
    for ax, mm in zip(axs, ms):
        pts = [(common.metric(mm, c, 'speedup'), common.metric(mm, c, 'delta'), i) for i, c in enumerate(common.CONFIGS)]
        pts = [p for p in pts if np.isfinite(p[0] + p[1])]
        xx = [p[0] for p in pts]; yy = [p[1] for p in pts]
        dx = max(.06, (max(xx) - min(xx)) * .15); dy = max(3, (max(yy) - min(yy)) * .12)
        ax.set_xlim(min(xx) - dx, max(xx) + dx); ax.set_ylim(min(yy) - dy, max(yy) + dy)
        ax.fill_between([1, ax.get_xlim()[1]], 0, ax.get_ylim()[1], color='#EAF3ED', zorder=0)
        ax.axvline(1, color='#777777', ls=':', lw=.7, zorder=1); ax.axhline(0, color='#777777', ls=':', lw=.7, zorder=1)
        for x, y, i in pts:
            ax.scatter(x, y, s=60 if i == 0 else 22, marker=markers[i], color=common.COLORS[common.FAMILY_IDS[i]],
                       edgecolors='white' if i else '#202020', linewidths=.4, zorder=5 if i == 0 else 4)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=3, min_n_ticks=2)); ax.yaxis.set_major_locator(MaxNLocator(nbins=4, min_n_ticks=3))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{x:g}'))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{x:+g}' if abs(x) > 1e-10 else '0'))
        ax.grid(alpha=.13, lw=.5)
    axs[0].set_ylabel('Success change (points)')
    for ax in axs: ax.set_xlabel('Speedup (×)', fontsize=7.5)
    labels = {1: ['Keep 50%', 'Keep 20%'], 2: ['Repeat 2', 'Repeat 4'], 3: ['Prune 1 layer', 'Prune 2 layers', 'Prune 4 layers'],
              4: ['Reuse strict', 'Reuse moderate', 'Reuse aggressive'], 5: ['Fusion conservative', 'Fusion motion/entropy', 'Fusion task-aware']}
    handles = [Line2D([], [], marker='*', color='#202020', ls='', markersize=7, label='Original')]
    for i in range(1, 14):
        f = common.FAMILY_IDS[i]; j = sum(1 for k in range(i) if common.FAMILY_IDS[k] == f)
        handles.append(Line2D([], [], marker=markers[i], ls='', color=common.COLORS[f], markersize=4, label=labels[f][j]))
    fig.legend(handles=handles, loc='outside lower center', ncol=7, frameon=False, handletextpad=.3, columnspacing=.8)
    common.save(fig, name)

tradeoff_row('tradeoff_bridge_row')
print('done')

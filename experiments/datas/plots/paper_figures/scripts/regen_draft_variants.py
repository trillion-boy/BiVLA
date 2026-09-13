"""Draft variants of two catalog figures for the main paper.

1. success_change_bridge_labeled: the mentor's success_change_bridge with a
   backbone name in the corner of each panel (text, not a title, so the
   catalog's no-title rule still holds).
2. ablation_depth_pruning_masked: the mentor's ablation_depth_pruning with
   the SmolVLA LIBERO speedup points at budgets two and four removed, since
   Setup IV-A reads SmolVLA latency only within the released lerobot stack
   and those two budgets ran under the reconstructed stack.

Times New Roman is unavailable off macOS, so these render with Liberation
Serif, which is metric compatible. Re-run the catalog scripts on the
mentor's machine for the final Times rendering."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import common
from common import data, metric, models, model_name, env_name, save, CONFIGS, LABELS, COLORS, FAMILY_IDS, ENVS


def style():
    for p in font_manager.findSystemFonts():
        if 'LiberationSerif' in p:
            font_manager.fontManager.addfont(p)
    plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Liberation Serif'], 'font.size': 8,
                         'axes.titlesize': 9, 'axes.labelsize': 8, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
                         'legend.fontsize': 7, 'pdf.fonttype': 42, 'ps.fonttype': 42, 'mathtext.fontset': 'stix',
                         'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': .6,
                         'lines.linewidth': 1.1, 'lines.markersize': 4, 'savefig.dpi': 600, 'figure.dpi': 140,
                         'axes.axisbelow': True})


def comparison_labeled(name, env, key):
    from matplotlib.ticker import MaxNLocator
    style(); ms = models(env)
    cols = 2 if len(ms) == 4 else 3; rows = int(np.ceil(len(ms) / cols))
    fig, axs = plt.subplots(rows, cols, figsize=(7.16, 3.6 * rows), sharex=True, sharey=True, layout='constrained', squeeze=False)
    y = np.arange(13, dtype=float)
    for boundary in [2, 4, 7, 10]: y[boundary:] += .6
    values = np.array([[metric(m, c, key) for c in CONFIGS[1:]] for m in ms])
    lo = min(0, float(np.nanmin(values))); hi = max(0, float(np.nanmax(values)))
    pad = max((hi - lo) * .09, 2); xlim = (lo - pad, hi + pad)
    envlabel = {'bridge': 'WidowX', 'fractal': 'Google robot', 'libero': 'LIBERO'}[env]
    for ax, m, v in zip(axs.flat, ms, values):
        ax.axvline(0, color='#484848', lw=.8, zorder=2)
        for i, value in enumerate(v):
            color = COLORS[FAMILY_IDS[i + 1]]
            if not np.isfinite(value):
                ax.text(.98, y[i], 'Incomplete coverage', transform=ax.get_yaxis_transform(), ha='right', va='center', fontsize=6.5, color='#666666', style='italic')
                continue
            ax.barh(y[i], value, height=.62, color=color, zorder=3)
            if abs(value) < .0001: ax.plot(0, y[i], marker='|', color=color, markersize=5, zorder=4)
        ax.set_yticks(y, LABELS[1:]); ax.tick_params(axis='y', length=0, pad=5)
        ax.set_ylim(y[-1] + .8, -.8); ax.set_xlim(*xlim)
        ax.spines['left'].set_visible(False); ax.grid(axis='x', color='#e5e5e5', lw=.5)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4)); ax.tick_params(axis='x', labelbottom=True)
        # backbone name as in-axes text (the catalog forbids titles)
        ax.text(.02, .985, f'{model_name(m)} / {envlabel}', transform=ax.transAxes, ha='left', va='top', fontsize=7.5, fontweight='bold', color='#222222', bbox=dict(boxstyle='round,pad=.2', fc='white', ec='none', alpha=.85), zorder=6)
    for ax in list(axs.flat)[len(ms):]: ax.set_visible(False)
    fig.supxlabel('Success change from original (percentage points)', fontsize=9)
    save(fig, name)


def ablation_masked(name, family, mask):
    style(); inds = [i for i, f in enumerate(FAMILY_IDS) if f == family]; cs = ['original'] + [CONFIGS[i] for i in inds]
    labs = ['Original'] + [LABELS[i].split(': ')[-1] for i in inds]
    fig, axs = plt.subplots(1, 2, figsize=(7.16, 3.5), layout='constrained')
    palette = plt.get_cmap('tab20')
    for j, m in enumerate(models()):
        for ax, key in zip(axs, ['delta', 'speedup']):
            vals = [metric(m, c, key) for c in cs]
            if key == 'speedup' and m in mask:
                vals = [np.nan if c in mask[m] else v for c, v in zip(cs, vals)]
            ax.plot(range(len(cs)), vals, marker=['o', 's', '^'][list(ENVS).index(env_name(m))], color=palette(j), label=f'{model_name(m)} / {env_name(m).replace("bridge", "WidowX").replace("fractal", "Google")}')
            ax.set_xticks(range(len(cs)), labs); ax.grid(alpha=.16)
    axs[0].set_ylabel('Success change (percentage points)'); axs[1].set_ylabel('Time-per-step speedup (×)')
    axs[0].axhline(0, c='#aaaaaa', lw=.6); axs[1].axhline(1, c='#aaaaaa', lw=.6)
    axs[0].set_xlabel('Removed decoder layers'); axs[1].set_xlabel('Removed decoder layers')
    fig.legend(*axs[0].get_legend_handles_labels(), loc='outside lower center', ncol=4, fontsize=6)
    save(fig, name)


if __name__ == '__main__':
    comparison_labeled('success_change_bridge_labeled', 'bridge', 'delta')
    ablation_masked('ablation_depth_pruning_masked', 3, {'smolvla_libero': {'depth_pruning2', 'depth_pruning4'}})
    print('done')

"""Page-1 teaser (professor, 2026-09-15): one small panel per WidowX backbone, one bar
per configuration (the original counted as a trick, so 6 tricks and 14 bars), success
rate on the y axis, in the style of a grouped-bar benchmark figure. Data and colors are
the mentor's common.py; panel titles are used here because the requested style has them."""
import sys
from pathlib import Path
sys.path.insert(0, '/home/user/BiVLA/experiments/datas/plots/paper_figures/scripts')
import numpy as np, matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
import common
# Settings in the order of Section IV-A: keep 0.2 then 0.5; repeat 2 then 4; one, two, four
# layers; strict, moderate, aggressive; motion-entropy, task-aware, conservative-adaptive.
ORDER = ['original', 'fixed_foveation_keep20', 'fixed_foveation_keep50', 'action_repeat2', 'action_repeat4',
         'depth_pruning1', 'depth_pruning2', 'depth_pruning4', 'guarded_reuse_strict', 'guarded_reuse_moderate',
         'guarded_reuse_aggressive', 'temporal_fusion_motion_entropy', 'temporal_fusion_task_aware',
         'temporal_fusion_conservative_adaptive']
FAM = [0, 1, 1, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5]
NAMES = {'fixed_foveation_keep20': 'Keep 20%', 'fixed_foveation_keep50': 'Keep 50%', 'action_repeat2': 'Repeat 2',
         'action_repeat4': 'Repeat 4', 'depth_pruning1': 'Prune 1 layer', 'depth_pruning2': 'Prune 2 layers',
         'depth_pruning4': 'Prune 4 layers', 'guarded_reuse_strict': 'Reuse strict', 'guarded_reuse_moderate': 'Reuse moderate',
         'guarded_reuse_aggressive': 'Reuse aggressive', 'temporal_fusion_motion_entropy': 'Fusion motion/entropy',
         'temporal_fusion_task_aware': 'Fusion task-aware', 'temporal_fusion_conservative_adaptive': 'Fusion conservative'}
HERE = Path(__file__).resolve().parent
for p in font_manager.findSystemFonts():
    if 'LiberationSerif' in p: font_manager.fontManager.addfont(p)
plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Liberation Serif'], 'font.size': 7,
    'axes.titlesize': 7.5, 'axes.labelsize': 7, 'ytick.labelsize': 6, 'legend.fontsize': 6.3,
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'axes.spines.top': False, 'axes.spines.right': False,
    'axes.linewidth': .5, 'axes.axisbelow': True})

def shade(hexcol, k):
    """k = 0, 1, 2 for the first, second, third setting: full color then two lighter tints."""
    r, g, b = (int(hexcol[i:i + 2], 16) / 255 for i in (1, 3, 5)); f = [1.0, .72, .46][k]
    return (1 - f * (1 - r), 1 - f * (1 - g), 1 - f * (1 - b))

def teaser(name, env='bridge', size=(3.5, 2.75), grid=(2, 3), legend='top'):
    """grid = (rows, columns) of the six panels: (2, 3) is the original layout, (3, 2) the
    taller variant Soumya asked for (2026-09-15) to match the reference figure's finish.
    legend = 'top' (two rows above the panels) or 'right' (one column beside them, which
    frees the height the top legend used; Soumya, 2026-09-15 evening)."""
    ms = common.models(env)
    fig, axs = plt.subplots(*grid, figsize=size, layout='constrained')
    fams = FAM
    for ax, mm in zip(axs.flat, ms):
        vals = [common.metric(mm, c, 'success') for c in ORDER]
        x = []; pos = 0.0
        for i, f in enumerate(fams):
            if i and f != fams[i - 1]: pos += .6
            x.append(pos); pos += 1
        k_in_family = [sum(1 for j in range(i) if fams[j] == fams[i]) for i in range(14)]
        cols = [common.COLORS[0] if i == 0 else shade(common.COLORS[fams[i]], k_in_family[i]) for i in range(14)]
        ax.bar(x, vals, width=.9, color=cols, edgecolor='none', zorder=3)
        for xi, v, col in zip(x, vals, cols):
            if v == 0: ax.text(xi, 0.6, '0', ha='center', va='bottom', fontsize=5, color=col, fontweight='bold', zorder=4)
        ax.axhline(vals[0], color='#444444', ls=':', lw=.6, zorder=2)
        lo, hi = np.nanmin(vals), np.nanmax(vals); pad = max(4, (hi - lo) * .12)
        ax.set_ylim(max(0, lo - pad), min(100, hi + pad))
        ax.set_xticks([]); ax.tick_params(axis='y', length=2, pad=1)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))
        ax.grid(axis='y', color='#e6e6e6', lw=.5, zorder=0)
        ax.set_title(common.model_name(mm), pad=2)
    for ax in axs[:, 0]: ax.set_ylabel('Success (%)')
    handles = [Patch(color=common.COLORS[f], label=common.FAMILIES[f]) for f in range(6)]
    if legend == 'right':
        fig.legend(handles=handles, loc='outside right center', ncol=1, frameon=False, handlelength=1.1, handletextpad=.4, labelspacing=.9, borderaxespad=.1)
    else:
        fig.legend(handles=handles, loc='outside upper center', ncol=3, frameon=False, handlelength=1.1, handletextpad=.4, columnspacing=1.0, borderaxespad=.1)
    for ext in ['pdf', 'png']:
        (HERE / ext).mkdir(exist_ok=True)
        fig.savefig(HERE / ext / f'{name}.{ext}', bbox_inches='tight', pad_inches=.03, dpi=300, facecolor='white')

teaser('teaser_bridge_bars')
teaser('teaser_bridge_bars_3x2', size=(3.5, 3.55), grid=(3, 2))
teaser('teaser_bridge_bars_3x2_rl', size=(3.5, 3.15), grid=(3, 2), legend='right')
print('done', common.FAMILIES)

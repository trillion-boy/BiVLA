"""cross_setting_consistency with all 13 combinations in every row.

The catalog build excluded openvla_libero/temporal_fusion_conservative_adaptive
because its Long suite had not been run. That run has since completed
(OpenVLA_Libero_Long.json, 45 of 100). Its per-episode file is not yet
available, so this script pools the three per-episode suites with the Long
summary to obtain the aggregate success, marks the cell complete, and
redraws the figure with the mentor's consistency() unchanged.
Times New Roman is unavailable off macOS, so the render uses Liberation
Serif, which is metric compatible."""
import json
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib import font_manager
import common


def style_fallback():
    for p in font_manager.findSystemFonts():
        if 'LiberationSerif' in p:
            font_manager.fontManager.addfont(p)
    plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Liberation Serif'], 'font.size': 8,
                         'axes.titlesize': 9, 'axes.labelsize': 8, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
                         'legend.fontsize': 7, 'pdf.fonttype': 42, 'ps.fonttype': 42, 'mathtext.fontset': 'stix',
                         'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': .6,
                         'lines.linewidth': 1.1, 'lines.markersize': 4, 'savefig.dpi': 600, 'figure.dpi': 140,
                         'axes.axisbelow': True})


common.style = style_fallback

groups, agg, suites, audit, allrows = common.data()
m, c = 'openvla_libero', 'temporal_fusion_conservative_adaptive'
long = json.loads((common.ROOT.parent / 'OpenVLA_Libero_Long.json').read_text())
assert long['config_name'] == c and long['task_suite_name'] == 'libero_10'
parts = [suites[(m, c, s)] for s in ('libero_goal', 'libero_object', 'libero_spatial')]
succ = sum(p['successes'] for p in parts) + long['successes']
n = sum(p['n'] for p in parts) + long['episodes']
agg[(m, c)]['successes'] = succ
agg[(m, c)]['n'] = n
agg[(m, c)]['success'] = 100 * succ / n
agg[(m, c)]['complete'] = True
print(f'{m}/{c}: {succ}/{n} = {100*succ/n:.2f}% vs original {agg[(m,"original")]["success"]:.2f}%')

common.consistency('cross_setting_consistency')
print('written cross_setting_consistency.pdf/.png')

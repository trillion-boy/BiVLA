"""Significance-aware companion to cross_setting_consistency.

For each intervention configuration, counts how many of the 13
model-environment settings show a SIGNIFICANT lower, a non-significant, or a
SIGNIFICANT higher success than the original policy. Significance is an exact
two-sided McNemar test (binomial on discordant pairs) on episodes matched by
(task, episode_index), p < 0.05. LIBERO suites are combined per backbone, so a
setting is one backbone-environment. Incomplete settings are excluded, so the
conservative-fusion row has 12 settings rather than 13. Figure and panel titles
are omitted."""
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binomtest
from common import data, models, CONFIGS, LABELS, save


def style():
    # Times New Roman is unavailable off macOS; use a serif fallback with the
    # same layout rcParams as common.style(). Re-render with the paper font later.
    plt.rcParams.update({'font.family': 'serif', 'font.size': 8, 'axes.titlesize': 9,
                         'axes.labelsize': 8, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
                         'legend.fontsize': 7, 'pdf.fonttype': 42, 'ps.fonttype': 42,
                         'mathtext.fontset': 'stix', 'axes.spines.top': False,
                         'axes.spines.right': False, 'axes.linewidth': .6,
                         'lines.linewidth': 1.1, 'lines.markersize': 4,
                         'savefig.dpi': 600, 'figure.dpi': 140, 'axes.axisbelow': True})


def mcnemar_class(m, c):
    groups, agg, *_ = data()
    a = agg.get((m, c), {})
    if not a.get('complete', False):
        return None
    orig = {(e['task'], e['episode_index']): bool(e['success']) for e in groups[m, 'original']}
    inter = {(e['task'], e['episode_index']): bool(e['success']) for e in groups[m, c]}
    b = sum(1 for k, s in inter.items() if k in orig and s and not orig[k])   # fail -> success
    d = sum(1 for k, s in inter.items() if k in orig and not s and orig[k])   # success -> fail
    delta = a['success'] - agg[m, 'original']['success']
    n = b + d
    p = 1.0 if n == 0 else binomtest(b, n, 0.5).pvalue
    if p >= 0.05:
        return 'ns'
    return 'up' if delta > 0 else 'down'


def consistency_significance(name):
    style()
    fig, ax = plt.subplots(figsize=(7.16, 3.8), layout='constrained')
    counts = []
    for c in CONFIGS[1:]:
        cls = [mcnemar_class(m, c) for m in models()]
        cls = [x for x in cls if x is not None]
        counts.append([cls.count('down'), cls.count('ns'), cls.count('up')])
    counts = np.array(counts)
    left = np.zeros(13)
    palette = [('Significant lower', '#C45A52'), ('No significant change', '#D8D8D8'), ('Significant higher', '#377D9C')]
    for j, (lab, col) in enumerate(palette):
        v = counts[:, j]
        ax.barh(range(13), v, left=left, color=col, label=lab, height=.72)
        for i, nval in enumerate(v):
            if nval:
                ax.text(left[i] + nval / 2, i, str(int(nval)), ha='center', va='center', fontsize=7,
                        color='white' if j != 1 else '#333333')
        left += v
    ax.set_yticks(range(13), LABELS[1:])
    ax.invert_yaxis()
    ax.set_xlabel('Number of model–environment settings (exact McNemar, $p<0.05$)')
    ax.set_xlim(0, 13.4)
    fig.legend(*ax.get_legend_handles_labels(), loc='outside lower center', ncol=3, fontsize=7)
    save(fig, name)


if __name__ == '__main__':
    consistency_significance('cross_setting_significance')
    # quick text audit
    print('config'.ljust(30), 'sigDown', 'ns', 'sigUp')
    for c in CONFIGS[1:]:
        cls = [mcnemar_class(m, c) for m in models()]
        cls = [x for x in cls if x is not None]
        print(c.ljust(30), cls.count('down'), cls.count('ns'), cls.count('up'))

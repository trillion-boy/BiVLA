#!/usr/bin/env python3
"""Figure 1 -- the grid as a signed heat map.

The figure the Introduction points at with \\ref{fig:grid}. Its job is to let
a reader see, before reading a word, that the same intervention moves in
opposite directions depending on the cell it was measured in.

Every number is recomputed here from the episode records through
build_grid_report.paired(), the same function the report and verify_all.py
use. Nothing is transcribed, so the figure cannot drift from the text.

A cell is the paired delta in success points against that cell's own
baseline, over the episodes present in both runs. Cells whose condition was
never run are left blank rather than drawn as zero, because "no effect" and
"not measured" must not look the same.

Output: paper/fig_grid.pdf (vector, for the paper) and paper/fig_grid.png.
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_grid_report import discover, paired, grid_family  # noqa: E402

# The colour scale is clipped here, and the caption has to say so. Three cells
# sit at -81.2, -69.8 and -40.0 while the median cell is 8.3 points. Scaling to
# the extremes would render every cell that carries an argument -- the +15.6
# against -17.8, the +18.8 against -19.3 -- as near-white, which is the one
# thing this figure exists to prevent. Clipped cells keep their printed number,
# so nothing is hidden, and the colourbar is drawn with arrowheads to show the
# scale runs past its ends.
CLIP = 30.0

_HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(_HERE, "paper")

# Rows are the conditions, in the order the Setup section introduces them:
# time axis, then vision, then compute. The baseline is not a row, since every
# cell is measured against it and would be a row of zeros.
ROWS = [
    ("action_repeat2",   "action repetition 2"),
    ("action_repeat4",   "action repetition 4"),
    ("foveate",          "visual foveation, log-polar"),
    ("foveate_blur",     "visual foveation, blur"),
    ("depth_prune1",     "depth pruning 1"),
    ("depth_prune2",     "depth pruning 2"),
    ("depth_prune4",     "depth pruning 4"),
]

# Columns are the five filled cells. UniVLA/Fractal is the empty sixth, since
# its public checkpoint is Bridge-only, and it is drawn as an empty column so
# the reader sees the grid is 3x2 with one gap rather than five arbitrary runs.
COLS = [
    ("OpenVLA",    "Bridge",  "OpenVLA\nBridge"),
    ("SpatialVLA", "Bridge",  "SpatialVLA\nBridge"),
    ("UniVLA",     "Bridge",  "UniVLA\nBridge"),
    ("OpenVLA",    "Fractal", "OpenVLA\nFractal"),
    ("SpatialVLA", "Fractal", "SpatialVLA\nFractal"),
    ("UniVLA",     "Fractal", "UniVLA\nFractal"),
]

# foveate and foveate_logpolar are the same condition under two run names, as
# CONDITION_ORDER in build_grid_report records. Try both before calling a cell
# unmeasured.
ALIASES = {"foveate": ["foveate_logpolar", "foveate"]}


def value(data, backbone, bench, cond):
    """-> (delta, p) for one cell, or None if the condition was not run."""
    for name in ALIASES.get(cond, [cond]):
        got = paired(data, backbone, bench, name)
        if got is not None:
            return got[0], got[3]
    return None


def main():
    data = discover()

    grid = np.full((len(ROWS), len(COLS)), np.nan)
    pval = np.full((len(ROWS), len(COLS)), np.nan)
    for i, (cond, _) in enumerate(ROWS):
        for j, (bb, bench, _) in enumerate(COLS):
            got = value(data, bb, bench, cond)
            if got is not None:
                grid[i, j], pval[i, j] = got

    measured = int(np.isfinite(grid).sum())
    lo, hi = np.nanmin(grid), np.nanmax(grid)
    clipped = int((np.abs(grid) > CLIP).sum())
    print(f"{measured} measured cells, range {lo:+.1f} to {hi:+.1f}, "
          f"{clipped} past the clip at +/-{CLIP:.0f}")

    # The Bonferroni denominator is the number of paired tests the grid runs,
    # counted rather than written down. It is larger than the 35 cells drawn
    # here, since two cells also ran depth prune 8 and one ran the combination,
    # and using the drawn count would make the threshold too generous.
    cols = sorted({(b, k) for b, k, _ in COLS
                   if any(x[0] == b and x[1] == k for x in data)})
    fam = grid_family(data, cols)
    alpha = 0.05 / max(1, fam)
    print(f"correction family {fam} tests, alpha {alpha:.4f}")

    # A diverging map centred on zero, so the sign is the first thing read, and
    # symmetric so that -20 and +20 are equally saturated. An asymmetric limit
    # would put a slope in the figure that is not in the data.
    #
    # Every colour is then blended toward white. RdBu_r ends in navy and maroon,
    # which are too dark for black text, and the previous version dealt with
    # that by switching the number to white on the dark cells. That made the
    # numbers look like two categories when they are one. Capping the darkness
    # instead lets every number be the same colour, and the hue ordering, which
    # is what the reader actually uses, survives the blend intact.
    limit = CLIP
    norm = mcolors.TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)
    shades = plt.get_cmap("RdBu_r")(np.linspace(0.0, 1.0, 256))
    shades[:, :3] = 1.0 - (1.0 - shades[:, :3]) * 0.62
    cmap = mcolors.ListedColormap(shades)

    fig, ax = plt.subplots(figsize=(7.0, 3.3))
    ax.imshow(np.ma.masked_invalid(np.clip(grid, -limit, limit)),
              cmap=cmap, norm=norm, aspect="auto", interpolation="nearest")

    # A column with nothing in it gets one label rather than seven. Repeating
    # "not run" down the column reads as seven separate omissions, when the
    # truth is one fact about the checkpoint.
    empty_cols = {j for j in range(len(COLS))
                  if not np.isfinite(grid[:, j]).any()}
    for j in empty_cols:
        ax.text(j, (len(ROWS) - 1) / 2, "no public\ncheckpoint",
                ha="center", va="center", fontsize=7.5, color="0.45",
                style="italic", linespacing=1.5)

    # Significance is a box around the cell, not an asterisk on the number and
    # not bold. An asterisk lengthens some numbers and not others, so a column
    # of values stops aligning and the reader sees ragged text before seeing
    # which cells survived correction. A box is read as a property of the cell,
    # which is what significance is, and it leaves every number identical.
    n_sig = 0
    for i in range(len(ROWS)):
        for j in range(len(COLS)):
            v = grid[i, j]
            if not np.isfinite(v):
                continue
            if pval[i, j] < alpha:
                n_sig += 1
                ax.add_patch(plt.Rectangle(
                    (j - 0.5, i - 0.5), 1, 1, fill=False,
                    edgecolor="0.15", lw=1.9, zorder=4))
            ax.text(j, i, f"{v:+.1f}", ha="center", va="center",
                    fontsize=8.5, color="black", zorder=5)
    print(f"{n_sig} cells clear the correction")

    ax.set_xticks(range(len(COLS)))
    ax.set_xticklabels([c[2] for c in COLS], fontsize=8)
    ax.set_yticks(range(len(ROWS)))
    ax.set_yticklabels([r[1] for r in ROWS], fontsize=8)
    ax.tick_params(length=0)
    for side in ax.spines.values():
        side.set_visible(False)

    # Rules between the two benchmarks and between the three intervention
    # families. They are grey, and lighter than the significance boxes, because
    # the figure now uses dark line for exactly one thing. Structure and
    # encoding drawn in the same ink would be read as the same kind of mark.
    ax.axvline(2.5, color="0.55", lw=1.1)
    for y in (1.5, 3.5):
        ax.axhline(y, color="0.55", lw=0.9)

    ax.set_xticks(np.arange(-0.5, len(COLS), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(ROWS), 1), minor=True)
    ax.grid(which="minor", color="white", lw=1.5)
    ax.tick_params(which="minor", length=0)
    ax.set_axisbelow(True)  # else the white gridlines cut the boxes

    # The box has to be decodable from the figure. A caption-only key means a
    # reader who looks before reading sees an unexplained mark, which is the
    # state the asterisk was in.
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, fill=False,
                                     edgecolor="0.15", lw=1.9)],
              labels=[f"clears Bonferroni, $\\alpha = 0.05/{fam}$"],
              loc="upper left", bbox_to_anchor=(0.0, -0.16), frameon=False,
              handlelength=1.4, handleheight=1.0, fontsize=7.5,
              borderpad=0.0, handletextpad=0.6)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cb = fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.015, extend="both")
    cb.set_label("success points against the cell's own baseline", fontsize=7.5)
    cb.ax.tick_params(labelsize=7)
    cb.outline.set_visible(False)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        path = os.path.join(OUT, f"fig_grid.{ext}")
        fig.savefig(path, dpi=200, bbox_inches="tight")
        print("wrote", path)


if __name__ == "__main__":
    main()

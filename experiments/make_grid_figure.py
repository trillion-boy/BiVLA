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

    # ---------------------------------------------------------------- style
    # The figure is set in the paper's own face. IEEEtran sets Times, and
    # Liberation Serif is metric-compatible with it, so a reader does not see
    # the figure switch typeface mid-page. Sizes are absolute: the figure is
    # saved at the width it is printed at, so 7.5 pt here is 7.5 pt on paper.
    #
    # pdf.fonttype 42 is not cosmetic. Matplotlib defaults to Type 3, and IEEE
    # PDF eXpress rejects Type 3 fonts outright, so the default would have
    # failed at submission rather than at review.
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Liberation Serif", "Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.linewidth": 0.5,
    })

    # A diverging map centred on zero, so the sign is read first, and symmetric
    # so that -20 and +20 are equally saturated. An asymmetric limit would put
    # a slope in the figure that is not in the data.
    #
    # The map is RdBu_r with both ends TRUNCATED rather than blended toward
    # white. Blending was the first attempt and it washed the whole figure out,
    # because it desaturates the middle as hard as the ends, and the middle is
    # where most cells sit. Truncation removes only the navy and maroon that
    # black text cannot sit on, and leaves a -10 cell as legible as before.
    #
    # 0.24 is measured, not chosen by eye. At that cut the extremes are
    # (0.38, 0.65, 0.81) and (0.89, 0.48, 0.38), whose contrast against black
    # text is 7.9:1 and 7.2:1. WCAG's AAA threshold for body text is 7:1, so
    # every number in the figure clears it and all of them can be black.
    TRUNC = 0.24
    limit = CLIP
    norm = mcolors.TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)
    cmap = mcolors.ListedColormap(
        plt.get_cmap("RdBu_r")(np.linspace(TRUNC, 1.0 - TRUNC, 256)))

    # 7.00 in is the \textwidth a figure* spans, which is what this project's
    # ieeeconf.cls actually gives. MEASURED, 2026-08-25, with \showthe
    # \textwidth in the real document: 505.89 pt, and a TeX point is 1/72.27
    # in, so 7.000 in exactly. This said 7.16 before, on the assumption that
    # the class was IEEEtran; the file list says ieeeconf, which ICRA and IROS
    # distribute, and its margins differ. Drawing at final size means nothing
    # is scaled in LaTeX and the point sizes below are the ones on the page.
    fig = plt.figure(figsize=(7.00, 3.05))
    ax = fig.add_axes([0.215, 0.235, 0.778, 0.745])
    ax.imshow(np.ma.masked_invalid(np.clip(grid, -limit, limit)),
              cmap=cmap, norm=norm, aspect="auto", interpolation="nearest")

    # A column with nothing in it gets one label rather than seven. Repeating
    # "not run" down the column reads as seven separate omissions, when the
    # truth is one fact about the checkpoint.
    empty_cols = {j for j in range(len(COLS))
                  if not np.isfinite(grid[:, j]).any()}
    for j in empty_cols:
        ax.text(j, (len(ROWS) - 1) / 2, "no public\ncheckpoint",
                ha="center", va="center", fontsize=7, color="0.5",
                style="italic", linespacing=1.6)

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
                    edgecolor="0.1", lw=1.1, zorder=4))
            # U+2212 MINUS, not the hyphen f"{:+.1f}" emits. The colourbar
            # ticks are mathtext and already use a real minus, so the hyphen
            # made the two disagree inside one figure. An exact zero carries
            # no sign, since "+0.0" claims a direction the number does not.
            label = "0.0" if v == 0 else f"{v:+.1f}".replace("-", "\u2212")
            ax.text(j, i, label, ha="center", va="center",
                    fontsize=7.5, color="black", zorder=5)
    print(f"{n_sig} cells clear the correction")

    ax.set_xticks(range(len(COLS)))
    ax.set_xticklabels([c[2] for c in COLS], fontsize=7.5, linespacing=1.35)
    ax.set_yticks(range(len(ROWS)))
    ax.set_yticklabels([r[1] for r in ROWS], fontsize=7.5)
    ax.tick_params(length=0, pad=3)
    for side in ax.spines.values():
        side.set_visible(False)

    # Rules between the two benchmarks and between the three intervention
    # families. They are grey, and lighter than the significance boxes, because
    # the figure uses dark line for exactly one thing. Structure and encoding
    # drawn in the same ink would be read as the same kind of mark.
    # The family rules stop at the last measured column instead of running to
    # the axes edge. Past it there are no cells to separate, so the line was
    # floating in the white space of the empty column and made the grid look
    # unfinished rather than three columns short of a full one.
    last = max(j for j in range(len(COLS)) if np.isfinite(grid[:, j]).any())
    ax.axvline(2.5, color="0.45", lw=0.9)
    for y in (1.5, 3.5):
        ax.plot([-0.5, last + 0.5], [y, y], color="0.45", lw=0.7,
                solid_capstyle="butt", zorder=3)

    ax.set_xticks(np.arange(-0.5, len(COLS), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(ROWS), 1), minor=True)
    ax.grid(which="minor", color="white", lw=1.0)
    ax.tick_params(which="minor", length=0)
    ax.set_axisbelow(True)  # else the white gridlines cut the boxes

    # The colourbar is small and horizontal, in the strip under the grid. A
    # tall bar on the right cost a full column of width to restate what every
    # cell already prints as a number. Here colour is the secondary cue and the
    # number is the value, so the key only has to fix the direction and the
    # end points.
    cax = fig.add_axes([0.615, 0.062, 0.150, 0.030])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal", extend="both",
                      extendfrac=0.09, ticks=[-limit, 0, limit])
    cb.ax.set_xticklabels([f"$-{limit:.0f}$", "0", f"$+{limit:.0f}$"])
    cb.ax.tick_params(labelsize=6.5, length=2, width=0.5, pad=1.5)
    cb.outline.set_linewidth(0.5)
    cb.outline.set_edgecolor("0.4")
    # "vs.\ the cell's own baseline" used to follow. The backslash was a LaTeX
    # inter-word space pasted into a matplotlib string, where it is a literal
    # backslash, and the phrase ran off the canvas besides. The caption already
    # says what the baseline is, so the key only names the unit.
    cax.text(1.10, 0.5, "success points", transform=cax.transAxes,
             ha="left", va="center", fontsize=6.5)

    # The box has to be decodable from the figure. A caption-only key means a
    # reader who looks before reading sees an unexplained mark, which is the
    # state the asterisk was in.
    kax = fig.add_axes([0.215, 0.062, 0.30, 0.030]); kax.axis("off")
    kax.add_patch(plt.Rectangle((0.0, 0.12), 0.075, 0.76, fill=False,
                                edgecolor="0.1", lw=1.1,
                                transform=kax.transAxes, clip_on=False))
    kax.text(0.10, 0.5, f"clears Bonferroni, $\\alpha = 0.05/{fam}$",
             transform=kax.transAxes, ha="left", va="center", fontsize=6.5)

    for ext in ("pdf", "png"):
        path = os.path.join(OUT, f"fig_grid.{ext}")
        fig.savefig(path, dpi=400)
        print("wrote", path)


if __name__ == "__main__":
    main()

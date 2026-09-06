#!/usr/bin/env python3
"""Three candidate overview figures for the paper (one will be Fig. 1).

All three put the five interventions of Section III on one VLA inference
loop and show the evaluation protocol of Section II. They differ in
organisation:

  A  pipeline with insertion points   the VLA loop left to right, each
                                       intervention plugged in where it acts
  B  axes by role                      the three resource axes as columns,
                                       control / candidate as rows
  C  one call after another            what one environment step costs under
                                       dense inference and each intervention

Vector PDF (Type 42 fonts, IEEE PDF eXpress safe) and a PNG preview.
Width 7.16 in = IEEE double column. Run from the repo root:

    python3 experiments/paper/make_overview.py

Writes experiments/paper/figures/overview_{A,B,C}.{pdf,png}.
Nothing here states a value that belongs in Setup; k, rho, m stay symbols.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle, Polygon

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "figures")
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "font.family": "Liberation Sans",
    "font.size": 7,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "text.color": "#1e1e1e",
})

W = 7.16  # in

# palette: controls grey, candidates coloured, pipeline neutral
INK = "#1e1e1e"
GREY = "#6b7079"
GREY_FILL = "#eceef1"
PIPE_FILL = "#f7f8fa"
PIPE_EDGE = "#3a3f47"
C_FUSION = "#1f8a70"
C_DEPTH = "#6f4fa3"
C_REUSE = "#d9731a"
C_CTRL = "#7a8089"
C_REF = "#2b6cb0"
LIGHT = {C_FUSION: "#dff2ec", C_DEPTH: "#e9e2f5", C_REUSE: "#fbe8d6", C_CTRL: "#e6e8eb", C_REF: "#dce9f7"}


def canvas(h):
    fig = plt.figure(figsize=(W, h))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, h)
    ax.axis("off")
    return fig, ax


def box(ax, x, y, w, h, text="", fill=PIPE_FILL, edge=PIPE_EDGE, lw=0.8, fs=7, bold=False,
        dashed=False, rounding=0.05, color=INK, sub=None, subfs=6):
    p = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={rounding}",
                       fc=fill, ec=edge, lw=lw, ls="--" if dashed else "-")
    ax.add_patch(p)
    if text:
        ty = y + h / 2 + (0.05 if sub else 0)
        ax.text(x + w / 2, ty, text, ha="center", va="center", fontsize=fs,
                fontweight="bold" if bold else "normal", color=color)
    if sub:
        ax.text(x + w / 2, y + h / 2 - 0.07, sub, ha="center", va="center", fontsize=subfs, color=GREY)
    return p


def arrow(ax, x0, y0, x1, y1, color=PIPE_EDGE, lw=0.8, dashed=False, ms=7, rad=0.0):
    a = FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=ms, lw=lw,
                        color=color, ls="--" if dashed else "-",
                        connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0)
    ax.add_patch(a)


def line(ax, x0, y0, x1, y1, color=GREY, lw=0.6, dashed=True):
    ax.plot([x0, x1], [y0, y1], color=color, lw=lw, ls="--" if dashed else "-", solid_capstyle="round")


# ----------------------------------------------------------------- icons ---
def icon_foveation(ax, x, y, s):
    """square frame, blurred outside, sharp disc inside"""
    ax.add_patch(Rectangle((x, y), s, s, fc="#c9ccd1", ec=PIPE_EDGE, lw=0.5))
    ax.add_patch(Circle((x + s / 2, y + s / 2), 0.36 * s, fc="#dfe2e6", ec="none"))
    ax.add_patch(Circle((x + s / 2, y + s / 2), 0.26 * s, fc="white", ec=PIPE_EDGE, lw=0.5))


def icon_fusion(ax, x, y, s, color=C_FUSION, ring=True, reused=None, flagged=True):
    """4x4 token grid, reused cells filled, fresh cells white, ring around a flagged cell.
    reused=() draws only the flagged cell; flagged=False draws only the reused cells."""
    n = 4
    c = s / n
    if reused is None:
        reused = {(0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 3), (3, 0), (3, 3), (3, 1)}
    for i in range(n):
        for j in range(n):
            fc = LIGHT[color] if (i, j) in reused else "white"
            ax.add_patch(Rectangle((x + j * c, y + (n - 1 - i) * c), c, c, fc=fc, ec=PIPE_EDGE, lw=0.4))
    # flagged patch (moving) and its protective ring
    if flagged:
        ax.add_patch(Rectangle((x + 2 * c, y + (n - 1 - 2) * c), c, c, fc=color, ec=PIPE_EDGE, lw=0.4))
    if ring:
        ax.add_patch(Rectangle((x + 1 * c, y + (n - 1 - 3) * c), 3 * c, 3 * c, fc="none", ec=color, lw=0.9))


def icon_depth(ax, x, y, s, color=C_DEPTH, n=8, removed=(3, 5), protected=(0, 1, 2, 7)):
    """decoder stack, removed layers dashed, protected layers shaded"""
    h = s / n
    for i in range(n):
        yy = y + i * h
        if i in removed:
            ax.add_patch(Rectangle((x, yy + 0.15 * h), s, 0.7 * h, fc="white", ec=color, lw=0.6, ls="--"))
            ax.plot([x + 0.15 * s, x + 0.85 * s], [yy + 0.5 * h, yy + 0.5 * h], color=color, lw=0.9)
        else:
            fc = LIGHT[color] if i in protected else "white"
            ax.add_patch(Rectangle((x, yy + 0.15 * h), s, 0.7 * h, fc=fc, ec=PIPE_EDGE, lw=0.5))


def icon_reuse(ax, x, y, s, color=C_REUSE, fs=5.4, fs_gate=5.6,
               labels=("all pass: reuse $a_{t-1}$", "any fails: dense call")):
    """gate diamond with two exits, drawn in a box of width 1.1 s and height s.
    Returns the two exit-label artists so callers can check their fit."""
    cx, cy = x + 0.34 * s, y + 0.5 * s
    d = 0.38 * s
    ax.add_patch(Polygon([(cx - d, cy), (cx, cy + d), (cx + d, cy), (cx, cy - d)], closed=True,
                         fc=LIGHT[color], ec=color, lw=0.8))
    ax.text(cx, cy, "gates", ha="center", va="center", fontsize=fs_gate, color=color)
    arrow(ax, cx + d, cy, cx + d + 0.30 * s, cy + 0.34 * s, color=color, lw=0.7, ms=5, rad=-0.25)
    arrow(ax, cx + d, cy, cx + d + 0.30 * s, cy - 0.34 * s, color=PIPE_EDGE, lw=0.7, ms=5, rad=0.25)
    t1 = ax.text(cx + d + 0.34 * s, cy + 0.34 * s, labels[0], ha="left", va="center", fontsize=fs, color=color)
    t2 = ax.text(cx + d + 0.34 * s, cy - 0.34 * s, labels[1], ha="left", va="center", fontsize=fs, color=INK)
    return t1, t2


def icon_repeat(ax, x, y, s, color=C_CTRL, fs=6, h=None, span="$k$ steps",
                labels=("call", "hold", "\u2026", "hold", "call")):
    """timeline of width s over the given labels, with a bracket under the
    first four boxes (one action and its holds). h: total height (defaults
    to 0.42 s, the original square-ish proportion)."""
    if h is None:
        h = 0.42 * s
    labels = list(labels)
    w = s / len(labels)
    bh = 0.55 * h            # box height
    yb = y + 0.40 * h        # box bottom
    for i, lab in enumerate(labels):
        xx = x + i * w
        if lab == "\u2026":
            ax.text(xx + w / 2, yb + bh / 2, lab, ha="center", va="center", fontsize=fs + 1, color=GREY)
            continue
        call = lab == "call"
        ax.add_patch(Rectangle((xx + 0.06 * w, yb), 0.88 * w, bh,
                               fc=(LIGHT[color] if call else "white"), ec=(color if call else PIPE_EDGE),
                               lw=0.6, ls=("-" if call else "--")))
        ax.text(xx + w / 2, yb + bh / 2, lab, ha="center", va="center", fontsize=fs, color=INK)
    # bracket under the first k steps
    ax.plot([x + 0.06 * w, x + 0.06 * w, x + 4 * w - 0.06 * w, x + 4 * w - 0.06 * w],
            [y + 0.32 * h, y + 0.24 * h, y + 0.24 * h, y + 0.32 * h], color=GREY, lw=0.6)
    ax.text(x + 2 * w, y + 0.08 * h, span, ha="center", va="center", fontsize=fs, color=GREY)


# ------------------------------------------------------------- candidate A ---
def candidate_A():
    H = 3.35
    fig, ax = canvas(H)
    # --- pipeline row ---
    py, ph = 2.38, 0.46
    bw = 0.86
    xs = {"obs": 0.18, "enc": 1.42, "dec": 2.66, "out": 3.90, "act": 5.14, "env": 6.20}
    box(ax, xs["obs"], py, bw, ph, "Observation $o_t$", sub="image, instruction")
    box(ax, xs["enc"], py, bw, ph, "Visual encoder", sub="uniform patch grid")
    # decoder as a stack
    box(ax, xs["dec"], py, bw, ph, "", sub=None)
    for i in range(5):
        ax.add_patch(Rectangle((xs["dec"] + 0.10, py + 0.07 + i * 0.066), bw - 0.2, 0.05, fc="white", ec=PIPE_EDGE, lw=0.4))
    ax.text(xs["dec"] + bw / 2, py + ph + 0.06, "Decoder layers", ha="center", va="bottom", fontsize=7)
    box(ax, xs["out"], py, bw, ph, "Output stage", sub="tokens or action head")
    box(ax, xs["act"], py, 0.78, ph, "Action $a_t$", sub="or a chunk")
    box(ax, xs["env"], py, 0.78, ph, "Environment", sub="one step")
    for a, b in [("obs", "enc"), ("enc", "dec"), ("dec", "out"), ("out", "act")]:
        arrow(ax, xs[a] + bw, py + ph / 2, xs[b], py + ph / 2)
    arrow(ax, xs["act"] + 0.78, py + ph / 2, xs["env"], py + ph / 2)
    # feedback loop above
    top = py + ph + 0.28
    ax.plot([xs["env"] + 0.39, xs["env"] + 0.39, xs["obs"] + bw / 2], [py + ph, top, top], color=PIPE_EDGE, lw=0.8)
    arrow(ax, xs["obs"] + bw / 2, top, xs["obs"] + bw / 2, py + ph + 0.01)
    ax.text((xs["obs"] + xs["env"]) / 2 + 0.4, top + 0.04, "next observation $o_{t+1}$. Dense inference makes one call per $m$ executed actions",
            ha="center", va="bottom", fontsize=6.2, color=GREY)
    ax.text(xs["enc"] + bw + 0.19, py + ph / 2 + 0.07, "tokens", ha="center", va="bottom", fontsize=5.5, color=GREY)

    # --- intervention cards ---
    cy, ch = 1.00, 1.12
    cw = 1.30
    cards = [
        # x, colour, role, title, icon, caption, attach x (pipeline)
        (0.10, C_CTRL, "control", "Foveation", "fov", "sharp disc of keep\nratio $\\rho$, blur outside,\ntoken count same", xs["obs"] + bw + 0.19),
        (1.52, C_FUSION, "candidate", "Temporal fusion", "fus", "patches low in motion,\nentropy, attention take\nthe $t-1$ token, ring\nprotected, capped,\nkeyframes bound drift", xs["enc"] + bw + 0.19),
        (2.94, C_DEPTH, "candidate", "Depth pruning", "dep", "lowest Block\nInfluence layers\nremoved, ends kept,\nnone adjacent, set\non a calibration split", xs["dec"] + bw / 2),
        (4.36, C_REUSE, "candidate", "Guarded reuse", "reu", "before the call: if image stable,\nactions agree, motion commanded,\nskip the call and emit $a_{t-1}$, capped", xs["act"] + 0.39),
        (5.78, C_CTRL, "control", "Action repeat", "rep", "hold each action $k$ steps,\nno gate", xs["env"] + 0.39),
    ]
    for x, col, role, title, ic, cap, ax_x in cards:
        box(ax, x, cy, cw, ch, fill="white", edge=col, lw=0.9, rounding=0.06)
        ax.add_patch(Rectangle((x, cy + ch - 0.20), cw, 0.20, fc=LIGHT[col], ec="none"))
        ax.text(x + 0.07, cy + ch - 0.10, title, ha="left", va="center", fontsize=7.2, fontweight="bold", color=col)
        ax.text(x + cw - 0.06, cy + ch - 0.10, role, ha="right", va="center", fontsize=5.6, color=col, style="italic")
        s = 0.40
        if ic == "fov":
            icon_foveation(ax, x + 0.09, cy + 0.25, s)
        elif ic == "fus":
            icon_fusion(ax, x + 0.09, cy + 0.25, s)
        elif ic == "dep":
            icon_depth(ax, x + 0.10, cy + 0.22, s)
        elif ic == "reu":
            icon_reuse(ax, x + 0.08, cy + 0.44, 0.34)
        elif ic == "rep":
            icon_repeat(ax, x + 0.08, cy + 0.36, 1.14)
        if ic in ("reu", "rep"):
            ax.text(x + 0.07, cy + 0.07, cap, ha="left", va="bottom", fontsize=5.3, color=INK, linespacing=1.15)
        else:
            ax.text(x + 0.56, cy + 0.45, cap, ha="left", va="center", fontsize=5.3, color=INK, linespacing=1.15)
        # connector to the pipeline
        line(ax, x + cw / 2, cy + ch, ax_x, py - 0.02, color=col, lw=0.7)
        ax.add_patch(Circle((ax_x, py - 0.02), 0.03, fc=col, ec="none"))

    # --- protocol strip ---
    sy, sh = 0.10, 0.60
    box(ax, 0.10, sy, W - 0.20, sh, fill="#fbfbfc", edge=GREY, lw=0.6, rounding=0.06)
    ax.text(0.20, sy + sh - 0.10, "Protocol", ha="left", va="center", fontsize=7, fontweight="bold", color=C_REF)
    cols = [
        (0.20, "Reference", "the released policy, weights frozen,\ndense inference under fused attention"),
        (1.95, "Matched episodes", "same seeds for every condition,\npaired comparison on each backbone"),
        (3.70, "Grid", "every backbone on each environment with a\nreleased checkpoint at the evaluated size"),
        (5.50, "Positive gate", "latency down, own signal cost included,\nor success up, the other within a margin"),
    ]
    for x, t, s in cols:
        ax.text(x, sy + 0.34, t, ha="left", va="center", fontsize=6.4, fontweight="bold", color=INK)
        ax.text(x, sy + 0.15, s, ha="left", va="center", fontsize=5.3, color=GREY, linespacing=1.15)
    # legend for the icon fills
    ax.text(0.10, cy - 0.10, "icons: light fill = token reused from $t-1$ or protected layer, dark = flagged patch, ring = protected neighbours, "
            "dashed struck = removed layer, grey = blurred", ha="left", va="center", fontsize=5.2, color=GREY)
    return fig


# ------------------------------------------------------------- candidate B ---
def candidate_B():
    H = 3.0
    fig, ax = canvas(H)
    colx = [0.16, 2.19, 4.22]
    colw = 1.90
    heads = ["1) When the policy runs", "2) What it is shown", "3) How much of the decoder runs"]
    subs = ["model calls per environment step", "pixels or visual tokens", "decoder layers per call"]
    for x, h, s in zip(colx, heads, subs):
        box(ax, x, 2.52, colw, 0.36, fill=GREY_FILL, edge="none", rounding=0.05)
        ax.text(x + 0.08, 2.76, h, ha="left", va="center", fontsize=7.2, fontweight="bold")
        ax.text(x + 0.08, 2.61, s, ha="left", va="center", fontsize=5.8, color=GREY)
    # row labels
    rows = [(1.42, 0.98, "Control", "no signal"), (0.12, 1.18, "Candidate", "acts only where a signal says it is safe")]
    for y, h, t, s in rows:
        ax.text(0.05, y + h / 2, t, ha="center", va="center", fontsize=6.2, fontweight="bold", color=GREY, rotation=90)

    def cell(x, y, h, col, title, ic, cap, empty=False):
        if empty:
            box(ax, x, y, colw, h, fill="white", edge=GREY, lw=0.6, dashed=True, rounding=0.06)
            ax.text(x + colw / 2, y + h / 2, cap, ha="center", va="center", fontsize=5.8, color=GREY, linespacing=1.2)
            return
        box(ax, x, y, colw, h, fill="white", edge=col, lw=0.9, rounding=0.06)
        ax.add_patch(Rectangle((x, y + h - 0.2), colw, 0.2, fc=LIGHT[col], ec="none"))
        ax.text(x + 0.07, y + h - 0.10, title, ha="left", va="center", fontsize=7, fontweight="bold", color=col)
        s = 0.40
        body_mid = y + (h - 0.2) / 2
        if ic == "fov":
            icon_foveation(ax, x + 0.10, body_mid - s / 2, s)
        elif ic == "fus":
            icon_fusion(ax, x + 0.10, body_mid - s / 2, s)
        elif ic == "dep":
            icon_depth(ax, x + 0.12, body_mid - s / 2, s)
        elif ic == "reu":
            icon_reuse(ax, x + 0.10, y + 0.46, 0.36)
        elif ic == "rep":
            icon_repeat(ax, x + 0.10, y + 0.20, 1.15)
        if ic in ("reu", "rep"):
            ax.text(x + 0.08, y + 0.08, cap, ha="left", va="bottom", fontsize=5.7, color=INK, linespacing=1.15)
        else:
            ax.text(x + 0.62, body_mid, cap, ha="left", va="center", fontsize=5.7, color=INK, linespacing=1.15)

    # controls row
    cell(colx[0], 1.42, 0.98, C_CTRL, "Action repeat", "rep", "hold each action $k$ steps, no gate")
    cell(colx[1], 1.42, 0.98, C_CTRL, "Foveation", "fov", "sharp disc of keep ratio $\\rho$,\nblur outside, token count same")
    cell(colx[2], 1.42, 0.98, None, "", "", "no control on this axis", empty=True)
    # candidates row
    cell(colx[0], 0.12, 1.18, C_REUSE, "Guarded reuse", "reu", "before the call: skip it and emit $a_{t-1}$\nonly when the image is stable, the last two\ndense actions agree and command motion, capped")
    cell(colx[1], 0.12, 1.18, C_FUSION, "Temporal fusion", "fus", "patches low in motion, entropy\nand optional attention take the\n$t-1$ token, ring, cap, keyframes")
    cell(colx[2], 0.12, 1.18, C_DEPTH, "Depth pruning", "dep", "lowest Block Influence layers\nremoved, ends kept, none adjacent,\nset on a calibration split")
    # right panel: protocol
    px = 6.22
    box(ax, px, 0.12, W - px - 0.10, 2.76, fill="#fbfbfc", edge=C_REF, lw=0.7, rounding=0.06)
    ax.text(px + 0.42, 2.70, "Protocol", ha="center", va="center", fontsize=7, fontweight="bold", color=C_REF)
    items = [
        ("Reference", "released policy,\nfused attention"),
        ("Grid", "every backbone on\neach environment\nwith a checkpoint"),
        ("Environments", "WidowX Bridge,\nGoogle Robot,\nLIBERO suites"),
        ("Episodes", "matched seeds,\npaired by seed"),
        ("Positive gate", "latency down, own\nsignal cost in, or\nsuccess up, other\nwithin a margin"),
    ]
    yy = 2.46
    for t, s in items:
        ax.text(px + 0.07, yy, t, ha="left", va="top", fontsize=6.2, fontweight="bold")
        ax.text(px + 0.07, yy - 0.13, s, ha="left", va="top", fontsize=5.5, color=GREY, linespacing=1.15)
        yy -= 0.45
    return fig


# ------------------------------------------------------------- candidate C ---
def candidate_C():
    H = 3.45
    fig, ax = canvas(H)
    lx = 0.10          # label column
    tx0, tx1 = 1.78, 5.30   # timeline span
    steps = 6
    sw = (tx1 - tx0) / steps
    rows = [
        ("Dense reference", C_REF, "one full call per $m$ executed actions", "dense"),
        ("Action repeat", C_CTRL, "control: call every $k$ steps,\nhold between", "repeat"),
        ("Guarded reuse", C_REUSE, "candidate: skip a call only\nwhen every gate passes", "reuse"),
        ("Depth pruning", C_DEPTH, "candidate: every call runs\nfewer decoder layers", "depth"),
        ("Temporal fusion", C_FUSION, "candidate: stable tokens from\n$t-1$, keyframes fully fresh", "fusion"),
        ("Foveation", C_CTRL, "control: same call, periphery\nblurred before the encoder", "fov"),
    ]
    rh = 0.40
    y0 = 2.92
    # header
    for i in range(steps):
        ax.text(tx0 + (i + 0.5) * sw, y0 + 0.12, f"$t{'+' + str(i) if i else ''}$", ha="center", va="bottom", fontsize=6, color=GREY)
    ax.text(tx0 + steps * sw / 2, y0 + 0.30, "environment steps", ha="center", va="bottom", fontsize=6.2, color=GREY)
    ax.text(5.42, y0 + 0.12, "what changes", ha="left", va="bottom", fontsize=6.2, color=GREY)
    for r, (name, col, desc, kind) in enumerate(rows):
        y = y0 - (r + 1) * rh - 0.02
        ax.text(lx, y + rh / 2 + 0.10, name, ha="left", va="center", fontsize=7, fontweight="bold", color=col)
        ax.text(lx, y + rh / 2 - 0.09, desc, ha="left", va="center", fontsize=5.3, color=GREY, linespacing=1.1)
        ax.plot([tx0, tx1], [y + 0.04, y + 0.04], color="#d0d3d8", lw=0.5)
        for i in range(steps):
            x = tx0 + i * sw + 0.06
            w = sw - 0.12
            full = 0.28
            if kind == "dense":
                ax.add_patch(Rectangle((x, y + 0.06), w, full, fc=LIGHT[C_REF], ec=C_REF, lw=0.6))
            elif kind == "repeat":
                if i % 2 == 0:
                    ax.add_patch(Rectangle((x, y + 0.06), w, full, fc=LIGHT[col], ec=col, lw=0.6))
                else:
                    ax.add_patch(Rectangle((x, y + 0.06), w, full, fc="white", ec=col, lw=0.6, ls="--"))
                    ax.text(x + w / 2, y + 0.06 + full / 2, "hold", ha="center", va="center", fontsize=6, color=col)
            elif kind == "reuse":
                skip = (i == 3)
                if skip:
                    ax.add_patch(Rectangle((x, y + 0.06), w, full, fc="white", ec=col, lw=0.6, ls="--"))
                    ax.text(x + w / 2, y + 0.06 + full / 2, "gates pass:\nreuse $a_{t-1}$", ha="center", va="center", fontsize=6, color=col, linespacing=1.0)
                else:
                    ax.add_patch(Rectangle((x, y + 0.06), w, full, fc=LIGHT[col], ec=col, lw=0.6))
                    ax.text(x + w / 2, y + 0.06 + full / 2, "gate fails:\ndense", ha="center", va="center", fontsize=6, color=INK, linespacing=1.0)
            elif kind == "depth":
                nl = 7
                lh = full / nl
                for j in range(nl):
                    removed = j in (2, 4)
                    ax.add_patch(Rectangle((x, y + 0.06 + j * lh), w, lh, fc=("white" if removed else LIGHT[col]),
                                           ec=col, lw=0.4, ls=("--" if removed else "-")))
            elif kind == "fusion":
                key = (i % 3 == 0)
                ax.add_patch(Rectangle((x, y + 0.06), w, full, fc=LIGHT[col] if key else "white", ec=col, lw=0.6))
                # token strip
                n = 6
                cw_ = w / n
                for j in range(n):
                    reused = (not key) and j in (0, 2, 5)
                    ax.add_patch(Rectangle((x + j * cw_, y + 0.06 + 0.16), cw_, 0.08, fc=(col if reused else "white"), ec=col, lw=0.3))
                ax.text(x + w / 2, y + 0.115, "keyframe" if key else "fused", ha="center", va="center", fontsize=6, color=col)
            elif kind == "fov":
                ax.add_patch(Rectangle((x, y + 0.06), w, full, fc=LIGHT[C_CTRL], ec=col, lw=0.6))
                icon_foveation(ax, x + w / 2 - 0.09, y + 0.11, 0.18)
        # right: what changes
        change = {
            "dense": "calls per step 1/$m$\nlayers all, tokens fresh",
            "repeat": "calls per step 1/$k$\nfeedback interval $\\times k$",
            "reuse": "calls per step below 1\nonly where gates pass",
            "depth": "layers per call fewer\nby the swept budget",
            "fusion": "some decoder-input tokens\nfrom $t-1$, same cost",
            "fov": "input detail reduced\nsame cost",
        }[kind]
        ax.text(5.42, y + rh / 2, change, ha="left", va="center", fontsize=5.5, color=INK, linespacing=1.15)
    # protocol strip
    sy, sh = 0.06, 0.30
    ax.text(lx, sy + sh / 2, "Every row runs on matched episodes against the dense reference under fused attention, on every backbone and environment of the grid. "
            "Repeat is drawn at $k$ = 2 and\nfusion with a keyframe every third step only to show the pattern. "
            "A candidate is positive if latency falls, its own signal cost included, or success rises, with the other within a margin.",
            ha="left", va="center", fontsize=5.6, color=INK, linespacing=1.2)
    return fig


if __name__ == "__main__":
    for name, fn in [("A", candidate_A), ("B", candidate_B), ("C", candidate_C)]:
        fig = fn()
        for ext in ("pdf", "png"):
            fig.savefig(os.path.join(OUT, f"overview_{name}.{ext}"), dpi=300 if ext == "png" else None)
        plt.close(fig)
        print("wrote", f"overview_{name}.pdf/png")


# ------------------------------------------------------------ candidate A2 ---
# A with the text moved to the caption: candidates above the loop and larger,
# controls below and smaller, a thin protocol line, and a thumbnail foveation.
import numpy as np


def foveation_thumb(n=64, keep=0.28, passes=10, k=5):
    """synthetic tabletop frame: sharp inside a disc, heavily box-blurred outside.

    passes x k: 10 passes of a 5x5 box give a blur sigma of about 4.5 px on a
    64 px frame, roughly twice the earlier 6 passes of 3x3 on 48 px, so the
    periphery still reads as blurred after the thumbnail is printed at 0.46 in.
    """
    yy, xx = np.mgrid[0:n, 0:n] / n
    img = 0.35 + 0.25 * yy                       # table gradient
    img[yy < 0.45] = 0.85 - 0.3 * yy[yy < 0.45]  # back wall
    # a few objects: a cube, a cylinder, a thin bar, a small dark part
    img[(xx > 0.22) & (xx < 0.40) & (yy > 0.50) & (yy < 0.68)] = 0.15
    img[np.hypot(xx - 0.66, yy - 0.62) < 0.10] = 0.95
    img[(xx > 0.05) & (xx < 0.95) & (yy > 0.78) & (yy < 0.82)] = 0.55
    img[np.hypot(xx - 0.50, yy - 0.40) < 0.05] = 0.05
    rng = np.random.default_rng(3)
    img = np.clip(img + 0.06 * rng.standard_normal((n, n)), 0, 1)
    blurred = img.copy()
    r = k // 2
    for _ in range(passes):
        pad = np.pad(blurred, r, mode="edge")
        blurred = sum(pad[i:i + n, j:j + n] for i in range(k) for j in range(k)) / float(k * k)
    rr = np.hypot(xx - 0.5, yy - 0.5)
    disc = rr <= np.sqrt(keep / np.pi)
    out = np.where(disc, img, blurred)
    return out, disc


def check_fit(fig, ax, items, tol=0.01):
    """items: (text artist, (x0, y0, x1, y1) in data units). Prints any text
    whose rendered extent leaves its box or the canvas. Returns the count."""
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    bad = 0
    W = ax.get_xlim()[1]
    Hh = ax.get_ylim()[1]
    for txt, (x0, y0, x1, y1) in items:
        bb = txt.get_window_extent(rend).transformed(ax.transData.inverted())
        if bb.x0 < x0 - tol or bb.x1 > x1 + tol or bb.y0 < y0 - tol or bb.y1 > y1 + tol:
            bad += 1
            print(f"OVERFLOW {txt.get_text()[:40]!r}: text [{bb.x0:.2f},{bb.x1:.2f}]x[{bb.y0:.2f},{bb.y1:.2f}]"
                  f" box [{x0:.2f},{x1:.2f}]x[{y0:.2f},{y1:.2f}]")
        if bb.x0 < -tol or bb.x1 > W + tol or bb.y0 < -tol or bb.y1 > Hh + tol:
            bad += 1
            print(f"OFF-CANVAS {txt.get_text()[:40]!r}")
    return bad


# Font floor for print (IEEE two-column, figure* at 7.16 in, no scaling):
# body text 7 pt, sub-text and icon labels 6.5 pt, card titles 8 pt.
FS_BODY, FS_SUB, FS_TITLE = 7.0, 6.5, 8.0


def candidate_A2(verbose=True):
    H = 3.45
    fig, ax = canvas(H)
    fit = []  # (text artist, allowed box) pairs for check_fit
    # --- pipeline row (middle) ---
    py, ph = 1.50, 0.46
    bw = 0.86
    xs = {"obs": 0.14, "enc": 1.32, "dec": 2.50, "out": 3.68, "act": 4.86, "env": 6.14}
    box(ax, xs["obs"], py, bw, ph, "Observation $o_t$", fs=FS_BODY, sub="image, instruction", subfs=FS_SUB)
    box(ax, xs["enc"], py, bw, ph, "Visual encoder", fs=FS_BODY, sub="patch tokens", subfs=FS_SUB)
    box(ax, xs["dec"], py, bw, ph, "")
    for i in range(5):
        ax.add_patch(Rectangle((xs["dec"] + 0.10, py + 0.06 + i * 0.068), bw - 0.2, 0.05, fc="white", ec=PIPE_EDGE, lw=0.4))
    t = ax.text(xs["dec"] + bw / 2, py - 0.045, "Decoder layers", ha="center", va="top", fontsize=FS_BODY, color=INK)
    box(ax, xs["out"], py, bw, ph, "Output stage", fs=FS_BODY, sub="tokens or head", subfs=FS_SUB)
    box(ax, xs["act"], py, bw, ph, "Action $a_t$", fs=FS_BODY, sub="$m$ per call", subfs=FS_SUB)
    box(ax, xs["env"], py, bw, ph, "Environment", fs=FS_BODY, sub="one step", subfs=FS_SUB)
    for a, b in [("obs", "enc"), ("enc", "dec"), ("dec", "out"), ("out", "act"), ("act", "env")]:
        arrow(ax, xs[a] + bw, py + ph / 2, xs[b], py + ph / 2)
    # feedback loop below the pipeline
    loop_y = py - 0.28
    ax.plot([xs["env"] + bw / 2, xs["env"] + bw / 2, xs["obs"] + bw / 2], [py, loop_y, loop_y], color=PIPE_EDGE, lw=0.8)
    arrow(ax, xs["obs"] + bw / 2, loop_y, xs["obs"] + bw / 2, py - 0.01)
    ax.text(3.6, loop_y - 0.04, "next observation $o_{t+1}$", ha="center", va="top", fontsize=FS_SUB, color=GREY)

    def card(x, y, w, h, col, title, role):
        box(ax, x, y, w, h, fill="white", edge=col, lw=1.0, rounding=0.06)
        ax.add_patch(Rectangle((x, y + h - 0.22), w, 0.22, fc=LIGHT[col], ec="none"))
        ax.text(x + 0.08, y + h - 0.11, title, ha="left", va="center", fontsize=FS_TITLE, fontweight="bold", color=col)
        ax.text(x + w - 0.07, y + h - 0.11, role, ha="right", va="center", fontsize=FS_SUB, color=col, style="italic")
        return (x, y, x + w, y + h - 0.22)  # body box for fit checks

    def keywords(x, y, lines, body, col=INK, fs=FS_BODY):
        t = ax.text(x, y, "\n".join(lines), ha="left", va="center", fontsize=fs, color=col, linespacing=1.3)
        fit.append((t, body))

    # --- candidates above (larger) ---
    cy, ch, cw = 2.30, 1.06, 2.28
    body_mid = cy + (ch - 0.22) / 2
    # temporal fusion, attached to the encoder->decoder edge
    x = 0.10
    body = card(x, cy, cw, ch, C_FUSION, "Temporal fusion", "candidate")
    icon_fusion(ax, x + 0.10, cy + 0.18, 0.48)
    keywords(x + 0.68, body_mid, ["motion, entropy, attention", "stable patches keep $t-1$ tokens", "capped, keyframes bound drift"], body)
    line(ax, x + cw / 2, cy, xs["enc"] + bw + 0.16, py + ph + 0.01, color=C_FUSION, lw=0.8)
    ax.add_patch(Circle((xs["enc"] + bw + 0.16, py + ph + 0.01), 0.03, fc=C_FUSION, ec="none"))
    # depth pruning, attached to the decoder
    x = 2.44
    body = card(x, cy, cw, ch, C_DEPTH, "Depth pruning", "candidate")
    icon_depth(ax, x + 0.12, cy + 0.16, 0.48)
    keywords(x + 0.72, body_mid, ["Block Influence ranking", "ends protected, none adjacent", "layers removed, not bypassed"], body)
    line(ax, x + cw / 2, cy, xs["dec"] + bw / 2, py + ph + 0.01, color=C_DEPTH, lw=0.8)
    ax.add_patch(Circle((xs["dec"] + bw / 2, py + ph + 0.01), 0.03, fc=C_DEPTH, ec="none"))
    # guarded reuse, attached to the action box
    x = 4.78
    body = card(x, cy, cw, ch, C_REUSE, "Guarded reuse", "candidate")
    for t in icon_reuse(ax, x + 0.10, cy + 0.40, 0.40, fs=FS_SUB, fs_gate=FS_SUB):
        fit.append((t, body))
    keywords(x + 0.08, cy + 0.19, ["image and action gates before the call", "pass: skip the call, reuse $a_{t-1}$"], body)
    line(ax, x + cw / 2, cy, xs["act"] + bw / 2, py + ph + 0.01, color=C_REUSE, lw=0.8)
    ax.add_patch(Circle((xs["act"] + bw / 2, py + ph + 0.01), 0.03, fc=C_REUSE, ec="none"))

    # --- controls below (smaller, grey) ---
    ky, kh, kw = 0.16, 0.92, 1.90
    x = 0.10
    body = card(x, ky, kw, kh, C_CTRL, "Foveation", "control")
    img, disc = foveation_thumb()
    ts = 0.48
    ext = [x + 0.09, x + 0.09 + ts, ky + 0.07, ky + 0.07 + ts]
    ax.imshow(img, cmap="gray", vmin=0, vmax=1, extent=ext, interpolation="bilinear", zorder=3)
    ax.add_patch(Circle(((ext[0] + ext[1]) / 2, (ext[2] + ext[3]) / 2), ts * np.sqrt(0.28 / np.pi), fc="none", ec="white", lw=0.7, zorder=4))
    ax.add_patch(Rectangle((ext[0], ext[2]), ts, ts, fc="none", ec=PIPE_EDGE, lw=0.5, zorder=4))
    keywords(x + 0.66, ky + (kh - 0.22) / 2, ["sharp disc, blurred", "periphery, same", "token count"], body)
    line(ax, x + kw / 2, ky + kh, xs["obs"] + bw + 0.16, py - 0.01, color=C_CTRL, lw=0.8)
    ax.add_patch(Circle((xs["obs"] + bw + 0.16, py - 0.01), 0.03, fc=C_CTRL, ec="none"))
    x = W - 0.10 - kw
    body = card(x, ky, kw, kh, C_CTRL, "Action repeat", "control")
    icon_repeat(ax, x + 0.10, ky + 0.22, kw - 0.20, fs=FS_SUB, h=kh - 0.22 - 0.26)
    t = ax.text(x + kw / 2, ky + 0.07, "hold each action $k$ steps, no gate", ha="center", va="bottom", fontsize=FS_SUB, color=GREY)
    fit.append((t, body))
    line(ax, x + kw / 2, ky + kh, xs["env"] + bw / 2, py - 0.01, color=C_CTRL, lw=0.8)
    ax.add_patch(Circle((xs["env"] + bw / 2, py - 0.01), 0.03, fc=C_CTRL, ec="none"))

    # --- protocol lines between the controls ---
    px0, px1 = 0.10 + kw + 0.18, W - 0.10 - kw - 0.18
    t = ax.text(px0, ky + kh - 0.02, "Protocol", ha="left", va="top", fontsize=FS_BODY + 0.5, fontweight="bold", color=C_REF)
    fit.append((t, (px0, ky, px1, ky + kh)))
    t = ax.text(px0, ky + kh - 0.19, "dense reference under fused attention, weights frozen\n"
                "matched episodes paired by seed, on every backbone\n"
                "and environment with a checkpoint\n"
                "positive only if latency falls, own signal cost included,\n"
                "or success rises, the other within a margin",
                ha="left", va="top", fontsize=FS_SUB, color=INK, linespacing=1.4)
    fit.append((t, (px0, ky, px1, ky + kh)))
    if verbose:
        n = check_fit(fig, ax, fit)
        print("fit check:", "clean" if n == 0 else f"{n} overflow(s)")
    return fig


# ------------------------------------------------------------ candidate A3 ---
# A2 with the slide-deck look removed: no protocol block (it lives in the
# caption and Section IV-A), no candidate/control tags (one legend line),
# one plain sentence per card instead of keyword fragments, and the foveation
# thumbnail is the real transform on the real Bridge scene.
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
# A real SimplerEnv WidowX Bridge observation (put eggplant in basket), the
# raw 640x480 frame captured in commit f9d54a0, with the empty Bridge
# background photograph from ManiSkill2_real2sim as a fallback.
FRAMES = [
    os.path.join(ROOT, "experiments", "figures", "obs_eggplant_raw.png"),
    os.path.join(ROOT, "SimplerEnv", "ManiSkill2_real2sim", "data", "real_inpainting", "bridge_real_eval_1.png"),
]


def foveation_real(keep_ratio=0.20, size=224):
    """The fixed-foveation transform as run in the MiniVLA harness
    (vla_tricks/foveation.py::foveate_blur on the tricks branch, applied to
    the 224 px policy input):
    sharp disc of area keep_ratio, then a blend into Gaussian sigma 3 and
    sigma 9 copies with distance. Applied to a real Bridge observation
    resized to the policy input size, as the harness does."""
    import cv2
    from PIL import Image
    path = next(p for p in FRAMES if os.path.exists(p))
    frame = np.asarray(Image.open(path).convert("RGB").resize((size, size), Image.BILINEAR), dtype=np.uint8)
    h, w = frame.shape[:2]
    cx, cy = w / 2.0, h / 2.0
    sharp_r = np.sqrt(keep_ratio * h * w / np.pi)
    max_r = np.hypot(max(cx, w - cx), max(cy, h - cy))
    ys, xs = np.mgrid[0:h, 0:w]
    dist = np.hypot(xs - cx, ys - cy)
    t = np.clip((dist - sharp_r) / max(max_r - sharp_r, 1e-6), 0, 1).astype(np.float32)
    middle = cv2.GaussianBlur(frame, (0, 0), sigmaX=3.0)
    far = cv2.GaussianBlur(frame, (0, 0), sigmaX=9.0)
    far_w = np.clip(2 * t - 1, 0, 1)[..., None]
    mid_w = np.clip(2 * t, 0, 1)[..., None] - far_w
    out = frame * (1 - mid_w - far_w) + middle * mid_w + far * far_w
    out = np.clip(np.rint(out), 0, 255).astype(np.uint8)
    out[dist <= sharp_r] = frame[dist <= sharp_r]
    return out, sharp_r / w


def candidate_A3(verbose=True):
    H = 3.16
    fig, ax = canvas(H)
    fit = []
    # --- pipeline row (middle) ---
    py, ph = 1.40, 0.46
    bw = 0.86
    xs = {"obs": 0.14, "enc": 1.32, "dec": 2.50, "out": 3.68, "act": 4.86, "env": 6.14}
    box(ax, xs["obs"], py, bw, ph, "Observation $o_t$", fs=FS_BODY, sub="image, instruction", subfs=FS_SUB)
    box(ax, xs["enc"], py, bw, ph, "Visual encoder", fs=FS_BODY, sub="patch tokens", subfs=FS_SUB)
    box(ax, xs["dec"], py, bw, ph, "")
    for i in range(5):
        ax.add_patch(Rectangle((xs["dec"] + 0.10, py + 0.17 + i * 0.054), bw - 0.2, 0.038, fc="white", ec=PIPE_EDGE, lw=0.4))
    ax.text(xs["dec"] + bw / 2, py + 0.075, "Decoder layers", ha="center", va="center", fontsize=FS_BODY, color=INK)
    box(ax, xs["out"], py, bw, ph, "Output stage", fs=FS_BODY, sub="tokens or values", subfs=FS_SUB)
    box(ax, xs["act"], py, bw, ph, "Action $a_t$", fs=FS_BODY, sub="$m$ per call", subfs=FS_SUB)
    box(ax, xs["env"], py, bw, ph, "Environment", fs=FS_BODY, sub="one step", subfs=FS_SUB)
    for a, b in [("obs", "enc"), ("enc", "dec"), ("dec", "out"), ("out", "act"), ("act", "env")]:
        arrow(ax, xs[a] + bw, py + ph / 2, xs[b], py + ph / 2)
    loop_y = py - 0.28
    ax.plot([xs["env"] + bw / 2, xs["env"] + bw / 2, xs["obs"] + bw / 2], [py, loop_y, loop_y], color=PIPE_EDGE, lw=0.8)
    arrow(ax, xs["obs"] + bw / 2, loop_y, xs["obs"] + bw / 2, py - 0.01)
    ax.text(3.6, loop_y - 0.04, "next observation $o_{t+1}$", ha="center", va="top", fontsize=FS_SUB, color=GREY)
    # guarded reuse: the gates branch off the observation edge and, when they
    # pass, hand a_{t-1} straight to the action box, so the model is not called
    gx, by = xs["obs"] + bw + 0.16, py - 0.13
    ax.plot([gx, gx, xs["act"] + bw / 2], [py + ph / 2, by, by], color=PIPE_EDGE, lw=0.8, ls="--")
    arrow(ax, xs["act"] + bw / 2, by, xs["act"] + bw / 2, py - 0.01, color=PIPE_EDGE)
    d = 0.05
    ax.add_patch(Polygon([(gx - d, py + ph / 2), (gx, py + ph / 2 + d), (gx + d, py + ph / 2), (gx, py + ph / 2 - d)],
                         closed=True, fc=LIGHT[C_REUSE], ec=C_REUSE, lw=0.8, zorder=5))
    ax.text(3.30, by - 0.07, "call skipped, $a_{t-1}$ repeated", ha="center", va="center", fontsize=FS_SUB, color=GREY)

    def card(x, y, w, h, col, title):
        box(ax, x, y, w, h, fill="white", edge=col, lw=1.0, rounding=0.06)
        ax.add_patch(Rectangle((x, y + h - 0.22), w, 0.22, fc=LIGHT[col], ec="none"))
        ax.text(x + 0.08, y + h - 0.11, title, ha="left", va="center", fontsize=FS_TITLE, fontweight="bold", color=col)
        return (x, y, x + w, y + h - 0.22)

    def sentence(x, y, text, body, col=INK):
        t = ax.text(x, y, text, ha="left", va="center", fontsize=FS_BODY, color=col, linespacing=1.3)
        fit.append((t, body))

    def attach(x0, y0, x1, y1, col):
        line(ax, x0, y0, x1, y1, color=col, lw=0.8)
        ax.add_patch(Circle((x1, y1), 0.032, fc=col, ec="white", lw=0.6, zorder=6))

    # --- candidates above ---
    cy, ch, cw = 2.16, 0.94, 2.28
    bm = cy + (ch - 0.22) / 2
    x = 0.10
    body = card(x, cy, cw, ch, C_FUSION, "Temporal fusion")
    icon_fusion(ax, x + 0.10, cy + 0.12, 0.48, ring=False, flagged=False)
    sentence(x + 0.68, bm, "Between keyframes a capped\nnumber of unprotected patches\nkeeps the previous call's token.", body)
    attach(x + cw / 2, cy, xs["enc"] + bw + 0.16, py + ph / 2, C_FUSION)
    x = 2.44
    body = card(x, cy, cw, ch, C_DEPTH, "Depth pruning")
    icon_depth(ax, x + 0.12, cy + 0.11, 0.48, protected=())
    sentence(x + 0.66, bm, "With the first layers and the last\nkept, the lowest in Block Influence\nare removed, none adjacent.", body)
    attach(x + cw / 2, cy, xs["dec"] + bw / 2, py + ph + 0.01, C_DEPTH)
    x = 4.78
    body = card(x, cy, cw, ch, C_REUSE, "Guarded reuse")
    for t in icon_reuse(ax, x + 0.06, cy + 0.15, 0.42, fs=FS_SUB, fs_gate=FS_SUB, labels=("pass", "fail")):
        fit.append((t, body))
    sentence(x + 0.76, bm, "When image and action gates\npass, the call is skipped and\n$a_{t-1}$ repeated, up to a cap.", body)
    attach(x + cw / 2, cy, xs["act"] + bw / 2, py + ph + 0.01, C_REUSE)

    # --- controls below, in the outer columns ---
    ky, kh, kw = 0.22, 0.80, 2.28
    x = 0.10
    body = card(x, ky, kw, kh, C_CTRL, "Foveation")
    img, r_frac = foveation_real()
    ts = 0.52
    ext = [x + 0.08, x + 0.08 + ts, ky + 0.03, ky + 0.03 + ts]
    ax.imshow(img, extent=ext, interpolation="bilinear", zorder=3)
    ax.add_patch(Circle(((ext[0] + ext[1]) / 2, (ext[2] + ext[3]) / 2), ts * r_frac, fc="none", ec="white", lw=0.6, zorder=4))
    ax.add_patch(Rectangle((ext[0], ext[2]), ts, ts, fc="none", ec=PIPE_EDGE, lw=0.5, zorder=4))
    sentence(x + 0.68, ky + (kh - 0.22) / 2, "The image is blurred outside\na central disc and keeps\nits token count.", body)
    attach(x + kw / 2, ky + kh, xs["obs"] + bw / 2 + 0.16, py - 0.01, C_CTRL)
    x = W - 0.10 - kw
    body = card(x, ky, kw, kh, C_CTRL, "Action repeat")
    icon_repeat(ax, x + 0.10, ky + 0.08, 1.30, fs=FS_SUB, h=0.46, span="$k$ steps", labels=("call", "hold", "\u2026", "hold"))
    sentence(x + 1.46, ky + (kh - 0.22) / 2, "Each action is\nheld $k$ steps, the\ncall skipped.", body)
    attach(x + kw / 2, ky + kh, 4.70, by, C_CTRL)

    # --- legend, one line ---
    t = ax.text(W / 2, 0.09, "Dots mark where each intervention enters the loop, the diamond marks the reuse gates, and the dashed path is a skipped call.",
                ha="center", va="center", fontsize=FS_SUB, color=GREY)
    fit.append((t, (0, 0, W, H)))
    if verbose:
        n = check_fit(fig, ax, fit)
        print("fit check:", "clean" if n == 0 else f"{n} overflow(s)")
    return fig


if __name__ == "__main__":
    for name, fn in (("A2", candidate_A2), ("A3", candidate_A3)):
        fig = fn()
        for ext in ("pdf", "png"):
            fig.savefig(os.path.join(OUT, f"overview_{name}.{ext}"), dpi=300 if ext == "png" else None)
        plt.close(fig)
        print(f"wrote overview_{name}.pdf/png")

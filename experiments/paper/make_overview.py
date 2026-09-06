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


def icon_fusion(ax, x, y, s, color=C_FUSION):
    """4x4 token grid, reused cells filled, fresh cells white, ring around a flagged cell"""
    n = 4
    c = s / n
    reused = {(0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 3), (3, 0), (3, 3), (3, 1)}
    for i in range(n):
        for j in range(n):
            fc = LIGHT[color] if (i, j) in reused else "white"
            ax.add_patch(Rectangle((x + j * c, y + (n - 1 - i) * c), c, c, fc=fc, ec=PIPE_EDGE, lw=0.4))
    # flagged patch (moving) and its protective ring
    ax.add_patch(Rectangle((x + 2 * c, y + (n - 1 - 2) * c), c, c, fc=color, ec=PIPE_EDGE, lw=0.4))
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


def icon_reuse(ax, x, y, s, color=C_REUSE):
    """gate diamond with two exits, drawn in a box of width 1.1 s and height s"""
    cx, cy = x + 0.34 * s, y + 0.5 * s
    d = 0.38 * s
    ax.add_patch(Polygon([(cx - d, cy), (cx, cy + d), (cx + d, cy), (cx, cy - d)], closed=True,
                         fc=LIGHT[color], ec=color, lw=0.8))
    ax.text(cx, cy, "gates", ha="center", va="center", fontsize=5.6, color=color)
    arrow(ax, cx + d, cy, cx + d + 0.30 * s, cy + 0.34 * s, color=color, lw=0.7, ms=5, rad=-0.25)
    arrow(ax, cx + d, cy, cx + d + 0.30 * s, cy - 0.34 * s, color=PIPE_EDGE, lw=0.7, ms=5, rad=0.25)
    ax.text(cx + d + 0.34 * s, cy + 0.34 * s, "all pass: reuse $a_{t-1}$", ha="left", va="center", fontsize=5.4, color=color)
    ax.text(cx + d + 0.34 * s, cy - 0.34 * s, "any fails: dense call", ha="left", va="center", fontsize=5.4, color=INK)


def icon_repeat(ax, x, y, s, color=C_CTRL):
    """timeline of width s: call, hold, ..., hold, call, with a bracket for k steps"""
    labels = ["call", "hold", "\u2026", "hold", "call"]
    w = s / len(labels)
    for i, lab in enumerate(labels):
        xx = x + i * w
        if lab == "\u2026":
            ax.text(xx + w / 2, y + 0.30 * s, lab, ha="center", va="center", fontsize=7, color=GREY)
            continue
        call = lab == "call"
        ax.add_patch(Rectangle((xx + 0.06 * w, y + 0.18 * s), 0.88 * w, 0.24 * s,
                               fc=(LIGHT[color] if call else "white"), ec=(color if call else PIPE_EDGE),
                               lw=0.6, ls=("-" if call else "--")))
        ax.text(xx + w / 2, y + 0.30 * s, lab, ha="center", va="center", fontsize=6, color=INK)
    # bracket under the first k steps
    ax.plot([x + 0.06 * w, x + 0.06 * w, x + 4 * w - 0.06 * w, x + 4 * w - 0.06 * w],
            [y + 0.14 * s, y + 0.10 * s, y + 0.10 * s, y + 0.14 * s], color=GREY, lw=0.6)
    ax.text(x + 2 * w, y + 0.02 * s, "$k$ steps", ha="center", va="center", fontsize=6, color=GREY)


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

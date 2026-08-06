"""Generate the two results figures from one table of measurements.

Written as a generator rather than hand-drawn SVG because the numbers are still
moving: every Delta here traces to a paired test in experiments/*.md, and when a
condition is re-run the figure has to follow without anyone re-typing a cell.

    python experiments/figures/make_result_figures.py

Writes figures/fig1_coverage.svg and figures/fig3_horizon.svg.

Evidence tiers are drawn, not implied. Half this grid is legacy evidence -- old
campaigns whose per-episode records were not kept, so no paired test is possible
and only the unpaired difference survives. A figure that renders those cells the
same as the paired-tested ones would overstate what we know, which is the exact
failure the re-measurement campaign existed to fix.
"""
import os

OUT = os.path.dirname(os.path.abspath(__file__))

# ── Ink ────────────────────────────────────────────────────────────────────
INK, MUT, FAINT = "#1a1a19", "#52514e", "#8a8a85"
RULE, SURF = "#d8d7d2", "#ffffff"
# Diverging: blue (gain) <-> red (loss), neutral gray midpoint. Steps 100-600 of
# the blue ramp; the red arm mirrors it. Magnitude is printed in every cell too,
# so color is a redundant channel rather than the only one.
BLUE = ["#cde2fb", "#86b6ef", "#2a78d6", "#184f95"]
RED = ["#f9dcdc", "#ea9a9a", "#d03b3b", "#8f2020"]
NEUTRAL = "#f0efec"
# Three-series categorical, all-pairs validated in light mode
# (worst CVD dE 9.2, worst normal-vision dE 24.0).
S = {"OpenVLA": "#2a78d6", "SpatialVLA": "#eb6834", "UniVLA": "#1baf7a"}


def band(d):
    """-> (fill, text) for a delta in success-rate points."""
    if d is None:
        return SURF, MUT
    a = abs(d)
    ramp = BLUE if d > 0 else RED
    i = 0 if a < 3 else 1 if a < 10 else 2 if a < 25 else 3
    if a < 3:
        return NEUTRAL, INK
    return ramp[i], ("#ffffff" if i >= 2 else INK)


# ── The measurements ───────────────────────────────────────────────────────
# tier: "bonf"   paired McNemar, clears Bonferroni for ~15 comparisons (a~0.003)
#       "p05"    paired McNemar, p < 0.05 but not 0.003
#       "ns"     paired McNemar, not distinguishable from chance
#       "legacy" earlier campaign, per-episode records not kept -> unpaired only
#       None     not run
# The last column is the SAME backbone as column 2 on a different benchmark.
# It is in the grid because that comparison is the point: if an intervention's
# sign can flip without changing the policy at all, no default is safe.
BACKBONES = [
    ("OpenVLA", 15.6, "Bridge\nno chunk, k=1"),
    ("SpatialVLA", 30.2, "Bridge\nchunk head, k~4"),
    ("UniVLA", 78.1, "Bridge\nnative chunk, k=5"),
    ("RoboVLMs", 39.6, "Bridge\nLSTM state, k=10"),
    ("SpatialVLA", 84.4, "Fractal\nsame policy as col 2"),
]
ROWS = [
    ("Fixed foveation", "log-polar, keep 20%", {
        "OpenVLA": (+18.8, "bonf"), "SpatialVLA": (-7.3, "legacy"),
        "UniVLA": (+8.3, "legacy"), "RoboVLMs": (-19.8, "legacy"),
        "SpatialVLA/Fractal": (None, None)}),
    ("Fixed foveation", "blur, keep 20%", {
        "OpenVLA": (+17.7, "bonf"), "SpatialVLA": (-2.1, "legacy"),
        "UniVLA": (-2.1, "legacy"), "RoboVLMs": (-16.7, "legacy"),
        "SpatialVLA/Fractal": (None, None)}),
    ("Action repeat", "hold for 2 env steps", {
        "OpenVLA": (-8.3, "ns"), "SpatialVLA": (+12.5, "p05"),
        "UniVLA": (-70.8, "bonf"), "RoboVLMs": (None, None),
        "SpatialVLA/Fractal": (+0.0, "ns")}),
    ("Action repeat", "hold for 4 env steps", {
        "OpenVLA": (-11.5, "bonf"), "SpatialVLA": (-12.5, "ns"),
        "UniVLA": (None, None), "RoboVLMs": (None, None),
        "SpatialVLA/Fractal": (-40.0, "bonf")}),
    ("Fixed depth pruning", "bypass redundant layers", {
        "OpenVLA": (+1.0, "ns"), "SpatialVLA": (-9.4, "legacy"),
        "UniVLA": (None, None), "RoboVLMs": (None, None),
        "SpatialVLA/Fractal": (None, None)}),
]

FINDINGS = [
    ("The same intervention reverses sign across backbones.",
     "Action repeat 2 costs UniVLA 70.8 points and gains SpatialVLA 12.5. "
     "Identical code, identical hook, opposite direction — not a difference of "
     "degree that a better default would smooth out."),
    ("It also changes sign without changing the policy at all.",
     "Columns 2 and 5 are one checkpoint on two benchmarks. Action repeat "
     "2 is +12.5 on Bridge and exactly +0.0 on Fractal (11 of 22 discordant "
     "pairs each way). So the benchmark decides too, not only the backbone."),
    ("Damage tracks distance from the trained execution length, not the horizon.",
     "OpenVLA (k=1) and UniVLA (k=5) are both already at their trained length and "
     "only lose by moving. SpatialVLA is the one backbone deployed below its own "
     "chunk, and it is the one that gains. See Figure 3."),
    ("Foveation is both the largest gain and the largest loss we measured.",
     "+18.8 on OpenVLA, −19.8 on RoboVLMs. The backbones that lose have "
     "something for it to break — explicit 3D position encoding, or a latent "
     "compression bottleneck; the ones that gain have neither."),
    ("Depth pruning looks free on the mean and is not free per task.",
     "OpenVLA's aggregate moves +1.0 (p≥0.80) while eggplant goes 25.0 → "
     "58.3 and the other three fall. Pruning buys decisiveness and spends "
     "precision; the mean hides the trade."),
]

TIER_NOTE = [
    ("bonf", "paired McNemar, clears Bonferroni (α≈0.003)"),
    ("p05", "paired McNemar, p < 0.05"),
    ("ns", "paired, not distinguishable from chance"),
    ("legacy", "earlier campaign — no per-episode records, unpaired only"),
]


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def glyph(x, y, tier):
    """Evidence tier as a shape, so the tier survives grayscale and CVD."""
    if tier == "bonf":       # filled square = strongest
        return (f'<rect x="{x-4}" y="{y-4}" width="8" height="8" fill="{INK}"/>')
    if tier == "p05":        # filled circle
        return f'<circle cx="{x}" cy="{y}" r="4" fill="{INK}"/>'
    if tier == "ns":         # hollow circle
        return (f'<circle cx="{x}" cy="{y}" r="3.6" fill="none" stroke="{MUT}" '
                f'stroke-width="1.4"/>')
    if tier == "legacy":     # open square, dashed = provisional
        return (f'<rect x="{x-4}" y="{y-4}" width="8" height="8" fill="none" '
                f'stroke="{MUT}" stroke-width="1.3" stroke-dasharray="2.2 1.8"/>')
    return ""


# ── Figure 1 ───────────────────────────────────────────────────────────────
def figure1():
    W = 1040
    LX, LW = 30, 232                      # row-label column
    CW = (W - 30 - LX - LW) / len(BACKBONES)
    HY, RH = 136, 62                      # header baseline, row height
    TOP = HY + 60                         # clears the two architecture lines
    MH = TOP + RH * len(ROWS)             # matrix bottom
    FY = MH + 92                          # findings block top
    FH = 82
    H = FY + FH * len(FINDINGS) + 30

    o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" '
         f'width="{W}" height="{H}" font-family="Helvetica Neue, Arial, sans-serif">',
         f'<rect width="{W}" height="{H}" fill="{SURF}"/>',
         f'<text x="{LX}" y="46" font-size="26" font-weight="600" fill="{INK}">'
         f'The same intervention, opposite answers</text>',
         f'<text x="{LX}" y="72" font-size="13.5" fill="{MUT}">'
         f'Change in success rate against each backbone’s own baseline.</text>',
         f'<text x="{LX}" y="90" font-size="13.5" fill="{MUT}">'
         f'Bridge columns: 4 tasks × 24 = 96 matched pairs per condition. '
         f'Fractal column: 135. Frozen weights throughout.</text>']

    # column headers
    for i, (name, base, arch) in enumerate(BACKBONES):
        cx = LX + LW + CW * i + CW / 2
        o.append(f'<text x="{cx:.0f}" y="{HY}" font-size="16" font-weight="600" '
                 f'fill="{INK}" text-anchor="middle">{name}</text>')
        o.append(f'<text x="{cx:.0f}" y="{HY+18}" font-size="11.5" fill="{MUT}" '
                 f'text-anchor="middle">baseline {base:.1f}%</text>')
        for j, ln in enumerate(arch.split("\n")):
            o.append(f'<text x="{cx:.0f}" y="{HY+33+j*12}" font-size="10" '
                     f'fill="{FAINT}" text-anchor="middle">{esc(ln)}</text>')

    o.append(f'<line x1="{LX}" y1="{TOP-8}" x2="{W-30}" y2="{TOP-8}" '
             f'stroke="{INK}" stroke-width="1.2"/>')

    prev = None
    for r, (group, detail, cells) in enumerate(ROWS):
        ry = TOP + RH * r
        if prev is not None and group != prev:
            o.append(f'<line x1="{LX}" y1="{ry}" x2="{W-30}" y2="{ry}" '
                     f'stroke="{RULE}" stroke-width="1"/>')
        if group != prev:
            o.append(f'<text x="{LX}" y="{ry+27}" font-size="14.5" '
                     f'font-weight="600" fill="{INK}">{esc(group)}</text>')
        o.append(f'<text x="{LX}" y="{ry+45}" font-size="11.5" fill="{MUT}">'
                 f'{esc(detail)}</text>')
        prev = group

        for i, (name, _, note) in enumerate(BACKBONES):
            # Two columns share the name "SpatialVLA"; the benchmark disambiguates.
            key = f"{name}/{note.split(chr(10))[0]}" if name == "SpatialVLA" and i > 1 else name
            d, tier = cells[key]
            x = LX + LW + CW * i
            if d is None:
                o.append(f'<text x="{x+CW/2:.0f}" y="{ry+RH/2+5:.0f}" font-size="13" '
                         f'fill="{FAINT}" text-anchor="middle">not run</text>')
                continue
            fill, txt = band(d)
            o.append(f'<rect x="{x+7:.0f}" y="{ry+7}" width="{CW-14:.0f}" '
                     f'height="{RH-14}" rx="4" fill="{fill}"/>')
            # U+2212 minus, not a hyphen -- applied to the number alone, since
            # a blanket replace would also hit `font-size` and `text-anchor`.
            val = f"{d:+.1f}".replace("-", "−")
            o.append(f'<text x="{x+CW/2:.0f}" y="{ry+RH/2+2:.0f}" font-size="20" '
                     f'font-weight="600" fill="{txt}" text-anchor="middle">'
                     f'{val}</text>')
            o.append(glyph(x + CW / 2, ry + RH - 17, tier))

    o.append(f'<line x1="{LX}" y1="{MH}" x2="{W-30}" y2="{MH}" '
             f'stroke="{INK}" stroke-width="1.2"/>')

    # evidence legend
    gx = LX
    for tier, note in TIER_NOTE:
        o.append(glyph(gx + 5, MH + 24, tier))
        o.append(f'<text x="{gx+16}" y="{MH+28}" font-size="11" fill="{MUT}">'
                 f'{esc(note)}</text>')
        gx += 24 + len(note) * 5.55

    # findings
    o.append(f'<text x="{LX}" y="{FY-22}" font-size="16" font-weight="600" '
             f'fill="{INK}">What the grid says</text>')
    for k, (head, body) in enumerate(FINDINGS):
        y = FY + FH * k
        o.append(f'<circle cx="{LX+11}" cy="{y+9}" r="11" fill="{INK}"/>')
        o.append(f'<text x="{LX+11}" y="{y+13.5}" font-size="12.5" '
                 f'font-weight="600" fill="#ffffff" text-anchor="middle">{k+1}</text>')
        o.append(f'<text x="{LX+32}" y="{y+13}" font-size="14" font-weight="600" '
                 f'fill="{INK}">{esc(head)}</text>')
        lines = wrap(body, 118)
        assert len(lines) <= 3, f"finding {k+1} runs {len(lines)} lines; box fits 3"
        for j, ln in enumerate(lines):
            o.append(f'<text x="{LX+32}" y="{y+33+j*16}" font-size="12.5" '
                     f'fill="{MUT}">{esc(ln)}</text>')
    o.append("</svg>")
    return "\n".join(o)


def wrap(text, n):
    out, line = [], ""
    for w in text.split():
        if len(line) + len(w) + 1 > n:
            out.append(line)
            line = w
        else:
            line = f"{line} {w}".strip()
    if line:
        out.append(line)
    return out


# ── Figure 3 ───────────────────────────────────────────────────────────────
# (horizon, success%) on Bridge. Horizon = env steps executed per model call.
# "chunk" is the action-chunk length the checkpoint was trained to emit, which
# is what panel B normalises by.
CURVES = [
    ("OpenVLA",    1, [(1, 15.6), (2, 7.3), (4, 4.2)]),
    ("SpatialVLA", 4, [(1, 30.2), (2, 42.7), (4, 17.7)]),
    ("UniVLA",     5, [(5, 78.1), (10, 7.3)]),
]
# Where each series name sits, per panel. Hand-placed rather than derived: with
# three curves crossing, an automatic rule puts labels on top of each other, and
# a collision in a paper figure is worse than a hard-coded offset.
#   (panel, series) -> (which point, dx, dy, anchor)
LABEL_AT = {
    ("A", "OpenVLA"):    (-1, 13, 5, "start"),
    ("A", "SpatialVLA"): (-1, 13, 5, "start"),
    ("A", "UniVLA"):     (0, -13, 5, "end"),
    ("B", "OpenVLA"):    (-1, 13, 4, "start"),
    ("B", "SpatialVLA"): (0, -13, 24, "end"),
    ("B", "UniVLA"):     (-1, 13, 5, "start"),
}


def figure3():
    from math import log10
    W, H = 1040, 600
    PT, PH, PW = 168, 246, 356
    PA, PB = 92, 92 + 356 + 134

    def sx(v, x0, lo, hi):
        return x0 + (log10(v) - log10(lo)) / (log10(hi) - log10(lo)) * PW

    o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" '
         f'width="{W}" height="{H}" font-family="Helvetica Neue, Arial, sans-serif">',
         f'<rect width="{W}" height="{H}" fill="{SURF}"/>',
         f'<text x="30" y="42" font-size="24" font-weight="600" fill="{INK}">'
         f'Each policy is best near the horizon it was trained to execute</text>',
         f'<text x="30" y="66" font-size="13" fill="{MUT}">'
         f'Open-loop horizon = environment steps executed per model call. Action '
         f'repeat is the only temporal operation that exists</text>',
         f'<text x="30" y="84" font-size="13" fill="{MUT}">'
         f'identically on every backbone, so it is what varies the horizon here. '
         f'SimplerEnv WidowX-Bridge, 96 episodes per point.</text>']

    # legend -- identity is never carried by color alone
    lx = 30
    for name, _, _ in CURVES:
        o.append(f'<line x1="{lx}" y1="112" x2="{lx+22}" y2="112" '
                 f'stroke="{S[name]}" stroke-width="2.4"/>')
        o.append(f'<circle cx="{lx+11}" cy="112" r="4.5" fill="{S[name]}"/>')
        o.append(f'<text x="{lx+29}" y="116" font-size="12.5" fill="{INK}">'
                 f'{name}</text>')
        lx += 40 + len(name) * 7.4

    def axes(x0, title, yticks, ylab, xticks, xlab, lo, hi, y_of):
        a = [f'<text x="{x0}" y="{PT-22}" font-size="14" font-weight="600" '
             f'fill="{INK}">{esc(title)}</text>']
        for g, strong in yticks:
            y = y_of(g)
            a.append(f'<line x1="{x0}" y1="{y:.1f}" x2="{x0+PW}" y2="{y:.1f}" '
                     f'stroke="{"#c3c2bd" if strong else RULE}" stroke-width="1"/>')
            a.append(f'<text x="{x0-9}" y="{y+4:.1f}" font-size="11" fill="{MUT}" '
                     f'text-anchor="end">{g}</text>')
        for t, lbl in xticks:
            a.append(f'<text x="{sx(t, x0, lo, hi):.1f}" y="{PT+PH+21}" '
                     f'font-size="11.5" fill="{MUT}" text-anchor="middle">{lbl}</text>')
        a.append(f'<text x="{x0+PW/2}" y="{PT+PH+43}" font-size="12" fill="{MUT}" '
                 f'text-anchor="middle">{esc(xlab)}</text>')
        a.append(f'<text x="{x0-58}" y="{PT+PH/2}" font-size="12" fill="{MUT}" '
                 f'text-anchor="middle" transform="rotate(-90 {x0-58} {PT+PH/2})">'
                 f'{esc(ylab)}</text>')
        return a

    # ── panel A: absolute horizon vs success ──
    loA, hiA = 1, 10

    def yA(v):
        return PT + PH - v / 85 * PH

    o += axes(PA, "A · absolute horizon",
              [(f"{g}%", False) for g in ()] or [("0%", False), ("20%", False),
                                                 ("40%", False), ("60%", False),
                                                 ("80%", False)],
              "success rate",
              [(1, "1"), (2, "2"), (4, "4"), (5, "5"), (10, "10")],
              "env steps per model call (log)", loA, hiA,
              lambda g: yA(float(g.rstrip("%"))))

    for name, chunk, pts in CURVES:
        col = S[name]
        xy = [(sx(h, PA, loA, hiA), yA(v)) for h, v in pts]
        tx = sx(chunk, PA, loA, hiA)
        o.append(f'<line x1="{tx:.1f}" y1="{PT}" x2="{tx:.1f}" y2="{PT+PH}" '
                 f'stroke="{col}" stroke-width="1" stroke-dasharray="3 3" '
                 f'opacity="0.45"/>')
        o.append(f'<polyline points="{" ".join(f"{x:.1f},{y:.1f}" for x, y in xy)}" '
                 f'fill="none" stroke="{col}" stroke-width="2"/>')
        for (x, y), (h, v) in zip(xy, pts):
            o.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5.5" fill="{col}" '
                     f'stroke="{SURF}" stroke-width="2"/>')
            above = v < 70          # keep the 78.1 label off the panel ceiling
            # A centred label on the leftmost point lands on the y-axis ticks.
            edge = x < PA + 14
            o.append(f'<text x="{x + (9 if edge else 0):.1f}" '
                     f'y="{y + (-13 if above else 21):.1f}" '
                     f'font-size="11.5" font-weight="600" fill="{INK}" '
                     f'text-anchor="{"start" if edge else "middle"}">{v:.1f}</text>')
        i, dx, dy, anc = LABEL_AT[("A", name)]
        o.append(f'<text x="{xy[i][0]+dx:.1f}" y="{xy[i][1]+dy:.1f}" '
                 f'font-size="12.5" font-weight="600" fill="{col}" '
                 f'text-anchor="{anc}">{name}</text>')

    # ── panel B: horizon relative to trained chunk ──
    loB, hiB = 0.2, 4.8

    def yB(d):
        return PT + PH - (d + 80) / 105 * PH

    o += axes(PB, "B · horizon ÷ trained chunk length",
              [("+20", False), ("0", True), ("−20", False), ("−40", False),
               ("−60", False), ("−80", False)],
              "change vs own baseline (points)",
              [(0.25, "0.25×"), (0.5, "0.5×"), (1, "1×"), (2, "2×"), (4, "4×")],
              "horizon as a multiple of the trained chunk length (log)",
              loB, hiB,
              lambda g: yB(float(g.replace("−", "-").replace("+", ""))))

    xr = sx(1, PB, loB, hiB)
    o.append(f'<line x1="{xr:.1f}" y1="{PT-4}" x2="{xr:.1f}" y2="{PT+PH}" '
             f'stroke="{INK}" stroke-width="1.2" stroke-dasharray="4 3"/>')
    o.append(f'<text x="{xr+7:.1f}" y="{PT+8}" font-size="11" fill="{INK}">'
             f'trained chunk length</text>')

    for name, chunk, pts in CURVES:
        col, base = S[name], pts[0][1]
        xy = [(sx(h / chunk, PB, loB, hiB), yB(v - base)) for h, v in pts]
        o.append(f'<polyline points="{" ".join(f"{x:.1f},{y:.1f}" for x, y in xy)}" '
                 f'fill="none" stroke="{col}" stroke-width="2"/>')
        for x, y in xy:
            o.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5.5" fill="{col}" '
                     f'stroke="{SURF}" stroke-width="2"/>')
        i, dx, dy, anc = LABEL_AT[("B", name)]
        o.append(f'<text x="{xy[i][0]+dx:.1f}" y="{xy[i][1]+dy:.1f}" '
                 f'font-size="12.5" font-weight="600" fill="{col}" '
                 f'text-anchor="{anc}">{name}</text>')

    # SpatialVLA's peak sits left of the line. Say so on the figure rather than
    # let the reader read the hypothesis as having come out clean.
    px, py = sx(0.5, PB, loB, hiB), yB(12.5)
    o.append(f'<path d="M{px:.1f},{py-11:.1f} L{px:.1f},{py-27:.1f}" '
             f'stroke="{MUT}" stroke-width="1"/>')
    o.append(f'<text x="{px:.1f}" y="{py-33:.1f}" font-size="11" fill="{MUT}" '
             f'text-anchor="middle">peak at 0.5×, not 1×</text>')

    cap = ("Two backbones peak exactly at their trained length; SpatialVLA peaks "
           "below it. Action repeat holds one action for k steps, which is not the "
           "same operation as executing a k-step chunk — so the x-axis is "
           "comparable across backbones, the mechanism behind it is not.")
    foot = ("In panel A the dashed vertical tick, drawn in each curve’s own "
            "colour, marks that policy’s trained chunk length. In panel B "
            "OpenVLA and UniVLA are deployed at 1× by construction, so both sit "
            "at 0 there; SpatialVLA is the only backbone with points on both "
            "sides of the line.")
    lines = wrap(cap, 132)
    y0 = H - 30 - 16 * (len(lines) + len(wrap(foot, 132)))
    for j, ln in enumerate(lines):
        o.append(f'<text x="30" y="{y0+j*16}" font-size="11.5" fill="{MUT}">'
                 f'{esc(ln)}</text>')
    for j, ln in enumerate(wrap(foot, 132)):
        o.append(f'<text x="30" y="{y0+(len(lines)+j)*16+8}" font-size="11" '
                 f'fill="{FAINT}">{esc(ln)}</text>')
    o.append("</svg>")
    return "\n".join(o)


for fname, svg in (("fig1_coverage.svg", figure1()),
                   ("fig3_horizon.svg", figure3())):
    p = os.path.join(OUT, fname)
    with open(p, "w") as fh:
        fh.write(svg)
    print("wrote", p)

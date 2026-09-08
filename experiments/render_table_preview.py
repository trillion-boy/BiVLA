#!/usr/bin/env python3
"""Render paper/tablesimpler.tex as a PNG for a visual check without LaTeX.

Parses the generated tabular body (the Overleaf format: Model, Parameters,
Policy, then six value cells with "{\\scriptsize\\textcolor{c}{(+d)}}"
changes) and draws it with matplotlib. Not part of the paper.
"""
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "paper", "tablesimpler.tex")
OUT = os.path.join(HERE, "paper", "figures", "tablesimpler_preview.png")

GREEN = "#2a7d2a"
RED = "#c0392b"
CELL = re.compile(r"([-\d.]+|--)(?:\s+\{\\scriptsize\\textcolor\{([^}]*)\}"
                  r"\{\(([^)]*)\)\}\})?")
MULTI = re.compile(r"\\multirow\[c\]\{\d+\}\{\*\}\{([^}]*)\}")


def parse():
    rows, model, params = [], "", ""
    for raw in open(SRC):
        line = raw.strip()
        m = MULTI.search(line)
        if m and line.startswith("\\multirow"):
            model = m.group(1)
            continue
        if m and line.startswith("& \\multirow"):
            params = m.group(1)
            continue
        if not line.endswith("\\\\") or not line.startswith("&") \
                or "Success" in line or "multicolumn" in line:
            continue
        parts = [c.strip() for c in line[:-2].split("&")]
        parts = [p for p in parts if p != ""]
        policy, cells = parts[0], parts[1:]
        rows.append((model, params, policy, cells))
        model, params = "", ""
    return rows


def split_cell(c):
    m = CELL.match(c)
    val, colour, chg = m.group(1), m.group(2), m.group(3)
    col = GREEN if colour and colour.startswith("green") else RED
    return val, chg, col


def main():
    rows = parse()
    n = len(rows)
    fig_h = 0.28 * (n + 3)
    fig = plt.figure(figsize=(16, fig_h))
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax.axis("off")
    heads = ["Model", "Params", "Policy", "Success (%)", "Latency (ms)",
             "Avg. Steps", "Success (%)", "Latency (ms)", "Avg. Steps"]
    xs = [0.01, 0.08, 0.13, 0.31, 0.45, 0.59, 0.73, 0.86, 0.99]
    y = 1.0
    dy = 1.0 / (n + 3)
    ax.text(0.45, y, "WidowX", ha="center", va="top", fontsize=9,
            fontweight="bold", transform=ax.transAxes)
    ax.text(0.86, y, "Google Fractal", ha="center", va="top", fontsize=9,
            fontweight="bold", transform=ax.transAxes)
    y -= dy
    for x, h in zip(xs, heads):
        ax.text(x, y, h, ha="left" if x < 0.2 else "right", va="top",
                fontsize=8, fontweight="bold", transform=ax.transAxes)
    y -= dy
    first = True
    for model, params, policy, cells in rows:
        if model and not first:
            ax.plot([0.01, 0.99], [y + 0.35 * dy] * 2, color="0.6", lw=0.5,
                    transform=ax.transAxes)
        first = False
        if model:
            ax.text(xs[0], y, model, ha="left", va="top", fontsize=8,
                    fontweight="bold", transform=ax.transAxes)
            ax.text(xs[1], y, params, ha="left", va="top", fontsize=8,
                    transform=ax.transAxes)
        ax.text(xs[2], y, policy, ha="left", va="top", fontsize=8,
                transform=ax.transAxes)
        for x, c in zip(xs[3:], cells):
            val, chg, col = split_cell(c)
            if chg is None:
                ax.text(x, y, val, ha="right", va="top", fontsize=8,
                        transform=ax.transAxes)
            else:
                ax.text(x - 0.05, y, val, ha="right", va="top", fontsize=8,
                        transform=ax.transAxes)
                ax.text(x, y, f"({chg})", ha="right", va="top", fontsize=7,
                        color=col, transform=ax.transAxes)
        y -= dy
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=160, bbox_inches="tight")
    print("wrote", os.path.relpath(OUT, os.path.dirname(HERE)))


if __name__ == "__main__":
    main()

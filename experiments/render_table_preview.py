#!/usr/bin/env python3
"""Render paper/tablesimpler.tex as a PNG for a visual check without LaTeX.

Parses the generated tabular body (rows of "& "-separated cells with \\up{}
and \\dn{} changes) and draws it with matplotlib. Not part of the paper.
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


def parse():
    rows = []
    for line in open(SRC):
        line = line.strip()
        if not line.endswith("\\\\") or line.startswith("\\multirow{2}") \
                or line.startswith("& & Success"):
            continue
        cells = [c.strip() for c in line[:-2].split("&")]
        m = re.search(r"\{90\}\{([^}]*)\}", cells[0])
        model = m.group(1) if m else ""
        rows.append((model, cells[1], cells[2:]))
    return rows


def split_cell(c):
    """Return (value, change, colour) from '87.50 \\up{+0.00}' or '--'."""
    m = re.match(r"([-\d.]+|--)(?:\s+\\(up|dn)\{([^}]*)\})?", c)
    val, kind, chg = m.group(1), m.group(2), m.group(3)
    col = GREEN if kind == "up" else RED if kind == "dn" else "black"
    return val, chg, col


def main():
    rows = parse()
    n = len(rows)
    fig_h = 0.28 * (n + 3)
    fig = plt.figure(figsize=(16, fig_h))
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax.axis("off")
    heads = ["Model", "Policy", "Success (%)", "Latency (ms)", "Avg. Steps",
             "Success (%)", "Latency (ms)", "Avg. Steps"]
    xs = [0.01, 0.10, 0.29, 0.43, 0.57, 0.71, 0.85, 0.99]
    y = 1.0
    dy = 1.0 / (n + 3)
    ax.text(0.43, y, "WidowX", ha="center", va="top", fontsize=9,
            fontweight="bold", transform=ax.transAxes)
    ax.text(0.85, y, "Google Fractal", ha="center", va="top", fontsize=9,
            fontweight="bold", transform=ax.transAxes)
    y -= dy
    for x, h in zip(xs, heads):
        ax.text(x, y, h, ha="left" if x < 0.2 else "right", va="top",
                fontsize=8, fontweight="bold", transform=ax.transAxes)
    y -= dy
    last_model = None
    for model, policy, cells in rows:
        if model and model != last_model and last_model is not None:
            ax.plot([0.01, 0.99], [y + 0.35 * dy] * 2, color="0.6", lw=0.5,
                    transform=ax.transAxes)
        if model:
            ax.text(xs[0], y, model, ha="left", va="top", fontsize=8,
                    fontweight="bold", transform=ax.transAxes)
            last_model = model
        ax.text(xs[1], y, policy, ha="left", va="top", fontsize=8,
                transform=ax.transAxes)
        for x, c in zip(xs[2:], cells):
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

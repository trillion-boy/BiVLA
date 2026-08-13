"""Render the four stages of `foveate_image_logpolar` for two keep values.

Report §4.3 (b) argues that `keep` controls only the subsample step while the
log-polar round-trip happens regardless. The figure shows that directly: at
keep=100% stage 3 is byte-identical to stage 2, so the only thing left is the
warp and its inverse -- yet the output still differs from the input, and the
difference sits on edges and in the periphery.

Stages match the code in adaptive_sparse_vla/foveation.py:

    warpPolar(LOG)  ->  subsample(keep)  ->  resize back  ->  warpPolar(INVERSE)

Usage
-----
    python experiments/make_logpolar_figure.py [out.png] [observation.png]

Defaults write experiments/figures/logpolar_stages.png from the committed
Bridge observation. The warp arguments are taken from the shared foveation
module rather than retyped, so the picture cannot drift from what the eval ran.
"""
from __future__ import annotations

import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, os.pardir))
from adaptive_sparse_vla.foveation import _uniform_sample_grid  # noqa: E402

FWD = cv2.INTER_LINEAR + cv2.WARP_FILL_OUTLIERS + cv2.WARP_POLAR_LOG
INV = FWD + cv2.WARP_INVERSE_MAP

CELL, PAD, TOP, LEFT = 260, 14, 46, 108
FONT, FS = cv2.FONT_HERSHEY_SIMPLEX, 0.44
COLS = ["1. input", "2. warpPolar (log)", "3. subsample (keep)",
        "4. inverse warp = output", "difference vs input"]


def warp_params(img):
    """Same center and maxRadius the shared module computes."""
    h, w = img.shape[:2]
    center = (w / 2.0, h / 2.0)
    max_radius = float(np.hypot(max(center[0], w - center[0]),
                                max(center[1], h - center[1])))
    return center, max_radius


def stages(img, keep):
    h, w = img.shape[:2]
    center, max_radius = warp_params(img)
    lp = cv2.warpPolar(img, (w, h), center, max_radius, FWD)
    ys, xs = _uniform_sample_grid(h, w, keep)
    sampled = lp[np.ix_(ys, xs)]
    interp = cv2.resize(sampled, (w, h), interpolation=cv2.INTER_LINEAR)
    restored = np.clip(cv2.warpPolar(interp, (w, h), center, max_radius, INV),
                       0, 255).astype(np.uint8)
    diff = np.abs(restored.astype(int) - img.astype(int)).max(axis=2)
    heat = cv2.applyColorMap(np.clip(diff * 4, 0, 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    return [img, lp, sampled, restored, heat], sampled.shape[:2]


def fit(im):
    """Letterbox a panel into CELL x CELL so panels of different sizes line up."""
    h, w = im.shape[:2]
    s = min(CELL / w, CELL / h)
    r = cv2.resize(im, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_NEAREST)
    out = np.full((CELL, CELL, 3), 40, np.uint8)
    y, x = (CELL - r.shape[0]) // 2, (CELL - r.shape[1]) // 2
    out[y:y + r.shape[0], x:x + r.shape[1]] = r
    cv2.rectangle(out, (x, y), (x + r.shape[1] - 1, y + r.shape[0] - 1), (90, 90, 90), 1)
    return out


def main(out_path, obs_path):
    img = cv2.imread(obs_path)
    if img is None:
        print(f"could not read {obs_path}")
        return 1
    h, w = img.shape[:2]
    rows = [(1.0, "keep = 100%"), (0.2, "keep = 20%")]

    canvas = np.full((TOP + len(rows) * (CELL + PAD + 26),
                      LEFT + len(COLS) * (CELL + PAD), 3), 24, np.uint8)
    for ci, name in enumerate(COLS):
        cv2.putText(canvas, name, (LEFT + ci * (CELL + PAD), 28), FONT, FS,
                    (235, 235, 235), 1, cv2.LINE_AA)

    for ri, (keep, label) in enumerate(rows):
        panels, shp = stages(img, keep)
        y0 = TOP + ri * (CELL + PAD + 26)
        cv2.putText(canvas, label, (8, y0 + CELL // 2), FONT, 0.5,
                    (120, 220, 255), 1, cv2.LINE_AA)
        caps = [f"{w}x{h}", f"{w}x{h}", f"{shp[1]}x{shp[0]}", f"{w}x{h}",
                "brighter = changed more"]
        for ci, (pan, cap) in enumerate(zip(panels, caps)):
            x0 = LEFT + ci * (CELL + PAD)
            canvas[y0:y0 + CELL, x0:x0 + CELL] = fit(pan)
            cv2.putText(canvas, cap, (x0, y0 + CELL + 18), FONT, 0.4,
                        (170, 170, 170), 1, cv2.LINE_AA)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    cv2.imwrite(out_path, canvas)
    print(f"wrote {out_path}  {canvas.shape[1]}x{canvas.shape[0]}")
    return 0


if __name__ == "__main__":
    a = sys.argv[1:]
    sys.exit(main(a[0] if a else os.path.join(HERE, "figures", "logpolar_stages.png"),
                  a[1] if len(a) > 1 else os.path.join(HERE, "figures", "obs_carrot_raw.png")))

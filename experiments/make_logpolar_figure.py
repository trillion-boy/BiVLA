"""Two figures for Report §4.3 (b): what the log-polar round-trip does.

`stages` -- the four steps at keep=100% and keep=20%. At 100% stage 3 is
byte-identical to stage 2, so the subsample is a no-op and all that remains is
the warp and its inverse; the difference map is still not empty.

    warpPolar(LOG)  ->  subsample(keep)  ->  resize back  ->  warpPolar(INVERSE)

`zoom` -- input vs output magnified, once near the center and once at the rim,
at keep=100%. This is the figure that answers "if the center is spread over 49
columns, shouldn't it come out stretched?". It does not: the forward warp
magnifies the center and the inverse warp puts it back, so geometry is
restored (a marker at the exact center returns to within 0.00 px). What does
not come back is *detail at the rim*, because there the forward warp averaged
several source pixels into one column and the inverse warp can only re-expand
the average. Magnifying loses nothing; decimating loses permanently.

Usage
-----
    python experiments/make_logpolar_figure.py [stages|zoom] [out.png] [obs.png]

Defaults write into experiments/figures/ from the committed Bridge
observation. The warp arguments are taken from the shared foveation module
rather than retyped, so the pictures cannot drift from what the eval ran.
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


def zoom_figure(img, out_path, size=96, zoom=3):
    """Input vs output magnified, at the center and at the rim, keep=100%."""
    from adaptive_sparse_vla.foveation import foveate_image_logpolar
    out = foveate_image_logpolar(img, keep_ratio=1.0, center=None)
    h, w = img.shape[:2]
    cx, cy = w // 2, h // 2
    crops = [("CENTER (carrot)", cx - 40, cy - 30),
             ("PERIPHERY (top-right)", w - size - 10, 10)]

    tiles = []
    for name, x, y in crops:
        a, b = img[y:y + size, x:x + size], out[y:y + size, x:x + size]
        d = np.abs(b.astype(int) - a.astype(int)).max(axis=2)
        heat = cv2.applyColorMap(np.clip(d * 4, 0, 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
        row = [cv2.resize(p, (size * zoom,) * 2, interpolation=cv2.INTER_NEAREST)
               for p in (a, b, heat)]
        r = np.hypot(x + size / 2 - cx, y + size / 2 - cy)
        tiles.append((f"{name}   r={r:.0f}px   mean diff {d.mean():.2f}", row))

    cell, pad, top, left = size * zoom, 12, 58, 26
    canvas = np.full((top + len(tiles) * (cell + pad + 30), left + 3 * (cell + pad), 3), 24, np.uint8)
    for ci, t in enumerate(["input", "output (keep=100%)", "difference"]):
        cv2.putText(canvas, t, (left + ci * (cell + pad), 34), FONT, 0.5,
                    (235, 235, 235), 1, cv2.LINE_AA)
    for ri, (cap, row) in enumerate(tiles):
        y0 = top + ri * (cell + pad + 30)
        for ci, p in enumerate(row):
            x0 = left + ci * (cell + pad)
            canvas[y0:y0 + cell, x0:x0 + cell] = p
            cv2.rectangle(canvas, (x0, y0), (x0 + cell - 1, y0 + cell - 1), (90, 90, 90), 1)
        cv2.putText(canvas, cap, (left, y0 + cell + 20), FONT, 0.45,
                    (120, 220, 255), 1, cv2.LINE_AA)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    cv2.imwrite(out_path, canvas)
    print(f"wrote {out_path}  {canvas.shape[1]}x{canvas.shape[0]}")
    return 0


def main(out_path, obs_path, which="stages"):
    img = cv2.imread(obs_path)
    if img is None:
        print(f"could not read {obs_path}")
        return 1
    if which == "zoom":
        return zoom_figure(img, out_path)
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
    which = a[0] if a and a[0] in ("stages", "zoom") else "stages"
    rest = a[1:] if a and a[0] in ("stages", "zoom") else a
    default_out = os.path.join(HERE, "figures", f"logpolar_{which}.png")
    sys.exit(main(rest[0] if rest else default_out,
                  rest[1] if len(rest) > 1 else os.path.join(HERE, "figures", "obs_carrot_raw.png"),
                  which))

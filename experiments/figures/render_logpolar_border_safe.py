"""Presentation-only renderer for the log-polar foveation figure.

Reuses the exact log-polar warp/downsample/upsample/inverse-warp pipeline
from adaptive_sparse_vla/foveation.py, but edge-replicate-pads the source
before the forward warp. This avoids the dark vignette that
WARP_FILL_OUTLIERS produces at the image border: max_radius is set by the
farthest *corner*, so along non-diagonal directions the true frame boundary
is closer than max_radius, and outlier pixels beyond it get filled with
black, which then bleeds into the real border during the downsample/upsample
interpolation.

This is cosmetic only -- it does not change any reported experiment number,
which used the bit-identical foveate_image_logpolar in foveation.py.

Usage: python render_logpolar_border_safe.py <src.png> <out.png> <keep_ratio>
"""
from __future__ import annotations

import math
import sys

import cv2
import numpy as np


def _uniform_sample_grid(height, width, keep_ratio):
    keep_ratio = float(np.clip(keep_ratio, 0.0, 1.0))
    scale = math.sqrt(keep_ratio)
    rows = max(1, int(round(height * scale)))
    cols = max(1, int(round(width * scale)))
    ys = np.linspace(0, height - 1, num=rows, dtype=np.int32)
    xs = np.linspace(0, width - 1, num=cols, dtype=np.int32)
    return ys, xs


def foveate_logpolar_border_safe(image, keep_ratio, center=None):
    frame = np.asarray(image, dtype=np.uint8)
    height, width = frame.shape[:2]
    if center is None:
        center = (width / 2.0, height / 2.0)
    max_radius = float(np.hypot(max(center[0], width - center[0]),
                                 max(center[1], height - center[1])))

    pad_l = int(math.ceil(max(0.0, max_radius - center[0])))
    pad_r = int(math.ceil(max(0.0, max_radius - (width - center[0]))))
    pad_t = int(math.ceil(max(0.0, max_radius - center[1])))
    pad_b = int(math.ceil(max(0.0, max_radius - (height - center[1]))))
    padded = cv2.copyMakeBorder(frame, pad_t, pad_b, pad_l, pad_r, cv2.BORDER_REPLICATE)
    pcenter = (center[0] + pad_l, center[1] + pad_t)

    forward_flags = cv2.INTER_LINEAR + cv2.WARP_POLAR_LOG
    inverse_flags = cv2.INTER_LINEAR + cv2.WARP_POLAR_LOG + cv2.WARP_INVERSE_MAP

    logpolar = cv2.warpPolar(padded, (width, height), pcenter, max_radius, forward_flags)
    ys, xs = _uniform_sample_grid(height, width, keep_ratio)
    sampled = logpolar[np.ix_(ys, xs)]
    interpolated = cv2.resize(sampled, (width, height), interpolation=cv2.INTER_LINEAR)
    restored = cv2.warpPolar(interpolated, (width, height), center, max_radius, inverse_flags)
    return np.asarray(np.clip(restored, 0, 255), dtype=np.uint8)


if __name__ == "__main__":
    src_path, out_path, keep_ratio = sys.argv[1], sys.argv[2], float(sys.argv[3])
    img = cv2.imread(src_path)
    if img is None:
        raise FileNotFoundError(src_path)
    out = foveate_logpolar_border_safe(img, keep_ratio=keep_ratio)
    cv2.imwrite(out_path, out)
    print(f"wrote {out_path}")

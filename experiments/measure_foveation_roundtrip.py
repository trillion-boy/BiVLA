"""How much does the log-polar round-trip change the image, per `keep`, per region?

Why this exists
---------------
Report §4.3 (b) argues that the foveation gain does not come from compression:
the largest gain (+30.2, p = 4.2e-7) is at `keep = 100%`, where *no sample is
discarded*. The argument rested on reading the code --

    warpPolar -> subsample(keep) -> resize -> warpPolar(INVERSE)

-- and noticing that `keep` only controls the subsample step, so the warp
round-trip happens unconditionally. That is a claim about pixels, and it was
never measured against pixels. This script measures it.

What it reports
---------------
Mean absolute pixel error (0-255 scale, averaged over channels) between the
input and its round-trip, split into:

  center      the middle half of each axis, i.e. 1/4 of the image area
  periphery   everything outside that

Two things should fall out, and both are what §4.3 (b) needs:

  1. At keep = 100% the error is *not zero*, and the periphery is degraded
     several times more than the center. So keep=100% is not "foveation off";
     it is already foveation, which is consistent with it being the best cell.
  2. As keep falls, the *center* degrades too. Lowering keep does not merely
     discard more periphery -- it damages the region the policy actually acts
     on, while saving ~0% compute (§4.3 a).

What it does not establish
--------------------------
The absolute magnitudes are image-dependent: a high-frequency synthetic grid
lands around 20-80 on this scale while a real Bridge observation lands around
1-8. Only the *direction* is stable across images, so quote the direction and
the ratio, not the absolute number, unless you also name the image.

Nothing here connects pixel error to success rate. That link is interpretation,
not measurement.

Usage
-----
    python experiments/measure_foveation_roundtrip.py                 # synthetic images
    python experiments/measure_foveation_roundtrip.py obs.png [...]   # plus real frames

The numbers quoted in Report §4.3 (b) come from a 640x480 Bridge observation,
which is the resolution the harness foveates at (the env's native output,
before the processor resizes).
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))
from adaptive_sparse_vla.foveation import foveate_image_logpolar  # noqa: E402

KEEPS = (1.0, 0.4, 0.2, 0.1)


def center_mask(h: int, w: int) -> np.ndarray:
    """Middle half of each axis -> 1/4 of the pixels."""
    m = np.zeros((h, w), bool)
    m[h // 4: 3 * h // 4, w // 4: 3 * w // 4] = True
    return m


def grid_image(h: int, w: int, step: int) -> np.ndarray:
    im = np.zeros((h, w, 3), np.uint8)
    im[::step, :, :] = 255
    im[:, ::step, :] = 255
    return im


def report(name: str, img: np.ndarray) -> None:
    h, w = img.shape[:2]
    c = center_mask(h, w)
    print(f"\n{name}  {w}x{h}   original pixels = {h * w:,}")
    print(f"{'keep':>6} {'samples after warp':>24} {'center':>8} {'periph':>8} {'ratio':>7}")
    for keep in KEEPS:
        out = foveate_image_logpolar(img, keep_ratio=keep, center=None)
        d = np.abs(out.astype(int) - img.astype(int)).mean(axis=2)
        a, b = d[c].mean(), d[~c].mean()
        # mirrors _uniform_sample_grid: each axis scales by sqrt(keep)
        sh, sw = max(1, round(h * np.sqrt(keep))), max(1, round(w * np.sqrt(keep)))
        print(f"{keep:>6.0%} {sh}x{sw} = {sh * sw:>13,} {a:>8.1f} {b:>8.1f} {b / a:>6.1f}x")


def main(paths: list[str]) -> int:
    rng = np.random.default_rng(0)
    images = {
        "grid/8 (synthetic)": grid_image(224, 224, 8),
        "grid/16 (synthetic)": grid_image(224, 224, 16),
        "uniform noise": rng.integers(0, 256, (224, 224, 3), dtype=np.uint8),
    }

    for p in paths:
        try:
            import cv2
            im = cv2.imread(p)
        except ImportError:
            print("cv2 unavailable; skipping image files")
            break
        if im is None:
            print(f"could not read {p}; skipping")
            continue
        images[os.path.basename(p)] = im

    print("log-polar round-trip error (mean |diff|, 0-255, averaged over channels)")
    print("center = middle 1/4 of the area; periphery = the rest")
    for name, img in images.items():
        report(name, img)

    print("\nkeep=100% keeps every sample yet still changes the image, and degrades")
    print("the periphery more than the center -- that is foveation, not 'foveation off'.")
    print("Lowering keep degrades the center as well, for ~0% compute saved.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

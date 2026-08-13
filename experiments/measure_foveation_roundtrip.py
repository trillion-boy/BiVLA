"""What the two foveation variants actually do to an image, measured.

Two questions, two reports.

  (1) round-trip error per `keep`      -- backs Report §4.3 (b)
  (2) radial detail curve per variant  -- backs RelatedWork §2.3 (b)

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

The radial curve
----------------
RelatedWork §2.3 (b) contrasts the two variants by how much detail survives at
each radius. That curve is *metric-dependent* in a way the prose used to hide:
on the same image at the same keep, dead-center retention reads 35% under
Laplacian energy and 75% under Sobel. So the second report prints both, and
the doc names which one it quotes.

The curve also shows why the two variants are not a matched pair: blur holds
the center bit-exactly and erases the far field, while log-polar degrades the
center and preserves the far field better. They remove different information in
different places, so their difference does not isolate "the geometry share".

What it does not establish
--------------------------
The absolute magnitudes are image-dependent: a high-frequency synthetic grid
lands around 20-80 on the error scale while a real Bridge observation lands
around 1-8. Only the *direction* is stable across images, so quote the direction
and the ratio, not the absolute number, unless you also name the image.

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
from adaptive_sparse_vla.foveation import (  # noqa: E402
    foveate_image_blur,
    foveate_image_logpolar,
)

KEEPS = (1.0, 0.4, 0.2, 0.1)
BANDS = ((0.0, 0.1), (0.2, 0.3), (0.4, 0.5), (0.5, 0.7), (0.7, 1.0))


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


def _energy(im, metric: str):
    """High-frequency energy map. cv2 is only needed for this half of the script."""
    import cv2
    g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).astype(np.float32)
    if metric == "Laplacian":
        return cv2.Laplacian(g, cv2.CV_32F, ksize=3) ** 2
    return (cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3) ** 2
            + cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3) ** 2)


def radial_curve(name: str, img: np.ndarray, keep: float = 0.2) -> None:
    """Detail surviving at each radius, for both variants, under both metrics."""
    h, w = img.shape[:2]
    ys, xs = np.mgrid[0:h, 0:w]
    r = np.hypot(xs - w / 2.0, ys - h / 2.0) / np.hypot(w / 2.0, h / 2.0)
    outs = {"log-polar": foveate_image_logpolar(img, keep_ratio=keep, center=None),
            "blur": foveate_image_blur(img, keep_ratio=keep, center=None)}

    print(f"\n{name}  {w}x{h}, keep={keep:.0%}")
    hdr = "".join(f"{f'{lo:.1f}-{hi:.1f}':>10}" for lo, hi in BANDS)
    print(f"  {'metric':>10} {'variant':>10}{hdr}")
    for metric in ("Laplacian", "Sobel"):
        E0 = _energy(img, metric)
        for vname, out in outs.items():
            E1 = _energy(out, metric)
            cells = "".join(
                f"{100 * E1[(r >= lo) & (r < hi)].sum() / E0[(r >= lo) & (r < hi)].sum():>9.0f}%"
                for lo, hi in BANDS)
            print(f"  {metric:>10} {vname:>10}{cells}")
    for vname, out in outs.items():
        print(f"  bit-exact pixels, {vname:>10}: "
              f"{100 * (out == img).all(axis=2).mean():.1f}%")
    print("  ^ same image, same keep: dead-center retention differs by metric.")


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

    print("=" * 72)
    print("(1) log-polar round-trip error  [Report 4.3 (b)]")
    print("    mean |diff|, 0-255, averaged over channels")
    print("    center = middle 1/4 of the area; periphery = the rest")
    print("=" * 72)
    for name, img in images.items():
        report(name, img)

    print("\nkeep=100% keeps every sample yet still changes the image, and degrades")
    print("the periphery more than the center -- that is foveation, not 'foveation off'.")
    print("Lowering keep degrades the center as well, for ~0% compute saved.")

    print("\n" + "=" * 72)
    print("(2) radial detail curve, both variants  [RelatedWork 2.3 (b)]")
    print("    retained high-frequency energy per radial band")
    print("=" * 72)
    try:
        import cv2  # noqa: F401
    except ImportError:
        print("cv2 unavailable; skipping the radial curve")
        return 0
    for name, img in images.items():
        radial_curve(name, img)

    print("\nblur holds the center bit-exactly and erases the far field; log-polar")
    print("degrades the center and preserves the far field better. Different")
    print("information removed in different places -- so subtracting the two does")
    print("NOT isolate the geometric-distortion share.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

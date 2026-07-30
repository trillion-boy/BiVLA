"""Pin `foveation.py` to the original it was ported from.

OpenVLA scores 0.0% under log-polar on `libero_spatial` against a 74.0%
baseline. A number that extreme is worth doubting, and the first thing to
doubt is the port: if the transform were subtly wrong, everything built on top
of it would be measuring an implementation bug rather than a property of the
policy.

So this compares our functions against the source they came from
(`RetinaBased/PythonProject/openvla_inference.py`, the foveated OpenVLA
evaluation the SimplerEnv results were produced with) and requires them to be
bit-identical, not merely similar. It also checks the one place the two
pipelines genuinely differ.

    python test_foveation_port.py
    # -> ALL FOVEATION PORT CHECKS PASSED

Notes on what this does and does not prove:

* Our `foveate_image_*` take an extra `center` argument the original lacks.
  With `center=None` they must reduce exactly to the original, which is what
  makes the SimplerEnv and LIBERO numbers comparable.
* Both apply foveation at the same point -- on the raw env frame, before the
  policy's own resize (original: `FoveatedOpenVLAInference.prepare_image`;
  ours: `eval_libero.apply_foveation`).
* The LIBERO path adds a JPEG round-trip + LANCZOS 224 that the SimplerEnv
  path does not, because OpenVLA's own LIBERO evaluation does (the fine-tuning
  images were stored as JPEG). That is an extra lossy step applied on top of a
  warped image, so it is checked here rather than assumed harmless.
"""
import io
import os
import sys

import numpy as np
from PIL import Image

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from foveation import foveate_image_blur, foveate_image_logpolar  # noqa: E402

ORIGINAL = os.path.join(
    os.path.dirname(_HERE), "RetinaBased", "PythonProject", "openvla_inference.py"
)

ok = 0


def check(label, cond, extra=""):
    global ok
    assert cond, f"FAIL: {label} {extra}"
    ok += 1
    print(f"  ok  {label} {extra}")


def load_original():
    """Pull just the two foveation functions out of the reference file.

    It imports transforms3d/simpler_env at module scope, which are not
    installed here and are irrelevant to the transform, so take the text
    above the first class and drop the unrelated imports.
    """
    src = open(ORIGINAL).read()
    head = src[: src.index("class ")]
    keep = [
        line for line in head.splitlines()
        if not line.startswith(("import ", "from "))
        or any(k in line for k in ("numpy", "cv2", "math", "typing"))
    ]
    ns = {}
    exec(compile("\n".join(keep), ORIGINAL, "exec"), ns)
    return ns["foveate_image_logpolar"], ns["foveate_image_blur"]


def psnr(a, b):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return float("inf") if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)


if not os.path.exists(ORIGINAL):
    print(f"reference not found at {ORIGINAL} -- skipping")
    raise SystemExit(0)

orig_logpolar, orig_blur = load_original()

print("[1] bit-identical to the original at center=None")
rng = np.random.default_rng(0)
for shape in [(256, 256, 3), (224, 224, 3), (480, 640, 3)]:
    for keep in (0.05, 0.2, 0.5, 1.0):
        img = rng.integers(0, 256, shape, dtype=np.uint8)
        check(f"logpolar {shape[:2]} keep={keep}",
              np.array_equal(orig_logpolar(img, keep_ratio=keep),
                             foveate_image_logpolar(img, keep_ratio=keep, center=None)))
        check(f"blur     {shape[:2]} keep={keep}",
              np.array_equal(orig_blur(img, keep_ratio=keep),
                             foveate_image_blur(img, keep_ratio=keep, center=None)))

print("\n[2] a supplied centre actually moves the fovea")
img = rng.integers(0, 256, (256, 256, 3), dtype=np.uint8)
check("off-centre blur differs from centred",
      not np.array_equal(foveate_image_blur(img, 0.2, center=None),
                         foveate_image_blur(img, 0.2, center=(40.0, 200.0))))
check("centre=(W/2,H/2) equals centre=None",
      np.array_equal(foveate_image_blur(img, 0.2, center=None),
                     foveate_image_blur(img, 0.2, center=(128.0, 128.0))))

print("\n[3] the LIBERO-only JPEG round-trip does not amplify the damage")
# Must be a real scene. JPEG's behaviour depends entirely on the image's
# frequency content, and a synthetic pattern gives the opposite answer to a
# rendered one -- a check that flips on substitute data is worse than no
# check, so this skips rather than guesses.
#
# The checked-in frame is SimplerEnv Bridge, not LIBERO, because the LIBERO
# assets cannot be fetched in the environment this was authored in. The
# conclusion transfers a fortiori: LIBERO's synthetic renders are smoother
# than Bridge's photographic texture, so they carry even less high-frequency
# detail for JPEG to lose. Swap in a LIBERO agentview frame (verify_oracle_gaze.py
# renders them) to measure it directly.
frame_path = os.path.join(_HERE, "assets", "bridge_agentview.png")
if not os.path.exists(frame_path):
    print(f"  -- no real frame at {frame_path}, skipping section 3")
    print(f"\nALL {ok} FOVEATION PORT CHECKS PASSED (section 3 skipped)")
    raise SystemExit(0)
frame = np.asarray(Image.open(frame_path).convert("RGB"), dtype=np.uint8)


def jpeg_resize(a, size=224):
    buf = io.BytesIO()
    Image.fromarray(a).save(buf, format="JPEG", quality=95)
    buf.seek(0)
    return np.asarray(Image.open(buf).convert("RGB").resize((size, size), Image.LANCZOS))


def plain_resize(a, size=224):
    return np.asarray(Image.fromarray(a).resize((size, size), Image.LANCZOS))


base_cost = psnr(jpeg_resize(frame), plain_resize(frame))
for label, arr in [("logpolar 20%", foveate_image_logpolar(frame, 0.20)),
                   ("blur 20%", foveate_image_blur(frame, 0.20))]:
    cost = psnr(jpeg_resize(arr), plain_resize(arr))
    # Higher PSNR = the JPEG step distorted it less. A foveated frame has less
    # high-frequency content left to lose, so the round-trip must not be
    # harsher on it than on the original -- otherwise OpenVLA's collapse could
    # be an artifact of LIBERO-specific preprocessing rather than of foveation.
    check(f"JPEG is no harsher on {label} than on the raw frame",
          cost >= base_cost - 0.5, f"{cost:.1f} dB vs {base_cost:.1f} dB raw")

lp = psnr(jpeg_resize(foveate_image_logpolar(frame, 0.20)), jpeg_resize(frame))
bl = psnr(jpeg_resize(foveate_image_blur(frame, 0.20)), jpeg_resize(frame))
check("log-polar removes more signal than blur at equal keep-percent",
      lp < bl, f"log-polar {lp:.1f} dB vs blur {bl:.1f} dB")

print(f"\nALL {ok} FOVEATION PORT CHECKS PASSED")

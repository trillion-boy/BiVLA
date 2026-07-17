"""Self-verification for the plug-in foveation upgrades.

Claims under test:
  1. `foveate_image_blur` preserves geometry (no pixel displacement) --
     the property that log-polar warping provably lacks -- while still
     degrading the periphery and keeping the fovea bit-identical.
  2. Both foveation functions accept a movable `center`, and `center=None`
     reproduces the original fixed-center behavior exactly (backward compat
     with every result measured so far).
  3. `MotionGaze` finds and tracks the moving region from frames alone, and
     holds gaze on static scenes.
"""
import numpy as np
import pytest

from foveation import MotionGaze, foveate_image_blur, foveate_image_logpolar


def _noise_image(h=256, w=256, seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)


def _sharpness(img, y0, y1, x0, x1):
    """Mean absolute horizontal gradient inside a window (high = sharp)."""
    win = img[y0:y1, x0:x1].astype(np.float32)
    return float(np.abs(np.diff(win, axis=1)).mean())


# ---------------------------------------------------------------- claim 1

def test_blur_does_not_displace_pixels():
    """A bright impulse must stay at its exact location under blur foveation
    (its energy may spread, but the peak cannot move) -- whereas the log-polar
    warp moves off-center content. This is the geometry-preservation contract
    that SpatialVLA's Ego3D back-projection needs."""
    img = np.zeros((256, 256, 3), dtype=np.uint8)
    img[40, 200] = 255  # far off-center impulse
    out = foveate_image_blur(img, 0.2)
    peak = np.unravel_index(np.argmax(out.sum(axis=2)), out.shape[:2])
    assert peak == (40, 200)

    warped = foveate_image_logpolar(img, 0.2)
    wpeak = np.unravel_index(np.argmax(warped.sum(axis=2)), warped.shape[:2])
    # Positive control: the warp really does displace the same impulse
    # (if it didn't, blur mode would be redundant).
    assert wpeak != (40, 200) or warped.max() == 0


def test_blur_fovea_bit_identical_periphery_degraded():
    img = _noise_image()
    out = foveate_image_blur(img, 0.2)
    h, w = img.shape[:2]
    cy, cx = h // 2, w // 2
    # fovea (well inside the sharp disc r0 = sqrt(0.2*H*W/pi) ~ 64px): exact
    assert np.array_equal(out[cy - 20:cy + 20, cx - 20:cx + 20],
                          img[cy - 20:cy + 20, cx - 20:cx + 20])
    # periphery (corner): most high-frequency energy removed
    sharp_corner_in = _sharpness(img, 0, 40, 0, 40)
    sharp_corner_out = _sharpness(out, 0, 40, 0, 40)
    assert sharp_corner_out < 0.5 * sharp_corner_in


def test_blur_keep_one_is_identity_and_shapes():
    img = _noise_image(120, 160, seed=1)
    assert np.array_equal(foveate_image_blur(img, 1.0), img)
    out = foveate_image_blur(img, 0.2)
    assert out.shape == img.shape and out.dtype == np.uint8


def test_blur_deterministic():
    img = _noise_image(seed=2)
    a = foveate_image_blur(img, 0.2)
    b = foveate_image_blur(img, 0.2)
    assert np.array_equal(a, b)


def test_blur_rejects_non_rgb():
    with pytest.raises(ValueError):
        foveate_image_blur(np.zeros((64, 64), dtype=np.uint8), 0.2)


# ---------------------------------------------------------------- claim 2

def test_blur_center_moves_the_sharp_region():
    img = _noise_image()
    out = foveate_image_blur(img, 0.1, center=(40.0, 40.0))  # fovea near top-left
    # near the requested center: bit-identical
    assert np.array_equal(out[30:50, 30:50], img[30:50, 30:50])
    # opposite corner: degraded
    assert _sharpness(out, 216, 256, 216, 256) < 0.5 * _sharpness(img, 216, 256, 216, 256)


def test_logpolar_center_none_matches_original_fixed_center():
    """center=None must reproduce the historical behavior bit-for-bit, so all
    previously measured logpolar results remain comparable."""
    img = _noise_image(seed=3)
    h, w = img.shape[:2]
    assert np.array_equal(
        foveate_image_logpolar(img, 0.2),
        foveate_image_logpolar(img, 0.2, center=(w / 2.0, h / 2.0)),
    )


def test_logpolar_center_moves_the_sharp_region():
    img = _noise_image()
    near = (48.0, 48.0)
    out = foveate_image_logpolar(img, 0.2, center=near)
    # Sharpness retained near the fovea should beat the far corner.
    s_near = _sharpness(out, 28, 68, 28, 68)
    s_far = _sharpness(out, 216, 256, 216, 256)
    assert s_near > 1.5 * s_far


# ---------------------------------------------------------------- claim 3

def test_motion_gaze_tracks_moving_square_and_holds_when_static():
    gaze = MotionGaze(alpha=0.8)
    h = w = 256
    base = _noise_image(h, w, seed=4) // 8  # low-contrast static background

    def frame_with_square(x, y):
        f = base.copy()
        f[y:y + 24, x:x + 24] = 255
        return f

    c0 = gaze.update(frame_with_square(20, 20))
    assert c0 == (w / 2.0, h / 2.0)  # first frame: no diff yet -> image center

    # square jumps to (180,180): the diff lights up both old and new sites,
    # so after a few frames of it moving locally there, gaze must be near it.
    for step in range(4):
        cx, cy = gaze.update(frame_with_square(180 + step * 2, 180))
    assert abs(cx - 190) < 40 and abs(cy - 190) < 40

    # static frames afterwards: gaze holds (no drift back to image center)
    held = gaze.update(frame_with_square(188, 180))
    held2 = gaze.update(frame_with_square(188, 180))
    assert held2 == held

    gaze.reset()
    again = gaze.update(frame_with_square(188, 180))
    assert again == (w / 2.0, h / 2.0)


def test_motion_gaze_center_feeds_foveation():
    """End-to-end plug-in path: gaze center -> blur foveation keeps the moving
    region sharp even though it is far off the image center."""
    gaze = MotionGaze(alpha=1.0)
    base = np.zeros((256, 256, 3), dtype=np.uint8)
    f1 = base.copy(); f1[190:214, 30:54] = 200
    f2 = base.copy(); f2[196:220, 36:60] = 200
    gaze.update(f1)
    center = gaze.update(f2)
    out = foveate_image_blur(_noise_image(seed=5), 0.15, center=center)
    cx, cy = int(center[0]), int(center[1])
    src = _noise_image(seed=5)
    assert np.array_equal(out[cy - 10:cy + 10, cx - 10:cx + 10],
                          src[cy - 10:cy + 10, cx - 10:cx + 10])

"""Tests for the log-polar foveation port (foveation.py)."""
import numpy as np
from foveation import _uniform_sample_grid, foveate_image_logpolar


def _test_image(height=256, width=256):
    """Synthetic scene: bright square at the center, gradient background."""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[:, :, 0] = np.linspace(0, 255, width, dtype=np.uint8)[None, :]
    img[:, :, 1] = np.linspace(0, 255, height, dtype=np.uint8)[:, None]
    ch, cw = height // 2, width // 2
    img[ch - 20:ch + 20, cw - 20:cw + 20] = 255
    return img


def test_output_shape_and_dtype():
    img = _test_image()
    out = foveate_image_logpolar(img, keep_ratio=0.2)
    assert out.shape == img.shape
    assert out.dtype == np.uint8
    print("ok: output preserves shape and dtype")


def test_sample_grid_area_matches_keep_ratio():
    ys, xs = _uniform_sample_grid(256, 256, keep_ratio=0.2)
    area = (len(ys) * len(xs)) / (256 * 256)
    assert abs(area - 0.2) < 0.02, f"retained area {area:.3f} != 0.2"
    print(f"ok: sample grid retains ~20% of pixels ({area:.1%})")


def test_center_sharper_than_periphery():
    """Foveation must degrade the periphery more than the fovea.

    Uses high-frequency noise: smooth regions survive subsampling anywhere,
    so only texture reveals where detail is actually retained.
    """
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, size=(256, 256, 3), dtype=np.uint8)
    out = foveate_image_logpolar(img, keep_ratio=0.2)
    h, w = img.shape[:2]
    ch, cw = h // 2, w // 2
    center_err = np.abs(
        out[ch - 30:ch + 30, cw - 30:cw + 30].astype(float)
        - img[ch - 30:ch + 30, cw - 30:cw + 30].astype(float)
    ).mean()
    corner_err = np.abs(
        out[:60, :60].astype(float) - img[:60, :60].astype(float)
    ).mean()
    assert center_err < corner_err, (
        f"center err {center_err:.2f} should be < corner err {corner_err:.2f}"
    )
    print(f"ok: center err {center_err:.2f} < periphery err {corner_err:.2f}")


def test_keep_zero_returns_black():
    img = _test_image()
    out = foveate_image_logpolar(img, keep_ratio=0.0)
    assert out.shape == img.shape and not out.any()
    print("ok: keep_ratio=0 -> black frame")


def test_rejects_non_rgb():
    try:
        foveate_image_logpolar(np.zeros((64, 64), dtype=np.uint8), keep_ratio=0.2)
    except ValueError:
        print("ok: non-HxWx3 input raises ValueError")
    else:
        raise AssertionError("expected ValueError for 2D input")


def test_deterministic():
    img = _test_image()
    a = foveate_image_logpolar(img, keep_ratio=0.2)
    b = foveate_image_logpolar(img, keep_ratio=0.2)
    assert np.array_equal(a, b)
    print("ok: transform is deterministic")


if __name__ == "__main__":
    test_output_shape_and_dtype()
    test_sample_grid_area_matches_keep_ratio()
    test_center_sharper_than_periphery()
    test_keep_zero_returns_black()
    test_rejects_non_rgb()
    test_deterministic()
    print("\nALL FOVEATION TESTS PASS")

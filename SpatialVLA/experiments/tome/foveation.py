"""
Foveation utilities for VLA observations (model- and env-agnostic plug-in).

`foveate_image_logpolar` is a direct port of the mentor's foveated-input
pathway from the RetinaBased OpenVLA evaluation
(RetinaBased/PythonProject/openvla_inference.py: _uniform_sample_grid +
foveate_image_logpolar), kept bit-identical (with `center=None`) so results
are comparable across backbones. It moves pixels (polar warp), which is fine
for coordinate-free backbones (OpenVLA) but corrupts the pixel-grid <-> 3D
correspondence that geometry-aware backbones rely on: SpatialVLA back-projects
each patch's grid coordinate through the camera intrinsics with ZoeDepth depth
(modeling_spatialvla.py: backproject_patch, Ego3DPositionEmbeddingMLP), so a
warped image stamps every token with a wrong 3D position.

`foveate_image_blur` is the geometry-preserving alternative: identical
"sharp fovea, degraded periphery" information profile, but implemented as a
space-variant blur -- NO pixel is displaced, so intrinsics-based
back-projection and depth estimation see a geometrically consistent image.

`MotionGaze` supplies a fovea center from plain frame differencing ("look
where things move" -- the gripper/object locus), needing nothing but the
frames themselves: no oracle state, no detector, no model access.

All entry points accept an optional `center=(cx, cy)` in pixel coordinates;
`None` reproduces the original fixed image-center behavior.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import cv2
import numpy as np


def _uniform_sample_grid(height: int, width: int, keep_ratio: float) -> Tuple[np.ndarray, np.ndarray]:
    keep_ratio = float(np.clip(keep_ratio, 0.0, 1.0))
    if keep_ratio <= 0.0:
        return np.array([0], dtype=np.int32), np.array([0], dtype=np.int32)

    sample_scale = math.sqrt(keep_ratio)
    sample_rows = max(1, int(round(height * sample_scale)))
    sample_cols = max(1, int(round(width * sample_scale)))
    ys = np.linspace(0, height - 1, num=sample_rows, dtype=np.int32)
    xs = np.linspace(0, width - 1, num=sample_cols, dtype=np.int32)
    return ys, xs


def foveate_image_logpolar(
    image: np.ndarray,
    keep_ratio: float,
    center: Optional[Tuple[float, float]] = None,
) -> np.ndarray:
    frame = np.asarray(image, dtype=np.uint8)
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(f"Expected HxWx3 image, got {frame.shape}")

    if keep_ratio <= 0.0:
        return np.zeros_like(frame)

    height, width = frame.shape[:2]
    if center is None:
        center = (width / 2.0, height / 2.0)
    else:
        center = (float(np.clip(center[0], 0, width - 1)),
                  float(np.clip(center[1], 0, height - 1)))
    max_radius = float(
        np.hypot(max(center[0], width - center[0]), max(center[1], height - center[1]))
    )
    forward_flags = cv2.INTER_LINEAR + cv2.WARP_FILL_OUTLIERS + cv2.WARP_POLAR_LOG
    inverse_flags = (
        cv2.INTER_LINEAR + cv2.WARP_FILL_OUTLIERS + cv2.WARP_POLAR_LOG + cv2.WARP_INVERSE_MAP
    )

    logpolar = cv2.warpPolar(frame, (width, height), center, max_radius, forward_flags)
    sample_ys, sample_xs = _uniform_sample_grid(height, width, keep_ratio)
    sampled = logpolar[np.ix_(sample_ys, sample_xs)]
    interpolated = cv2.resize(sampled, (width, height), interpolation=cv2.INTER_LINEAR)
    restored = cv2.warpPolar(interpolated, (width, height), center, max_radius, inverse_flags)
    return np.asarray(np.clip(restored, 0, 255), dtype=np.uint8)


def foveate_image_blur(
    image: np.ndarray,
    keep_ratio: float,
    center: Optional[Tuple[float, float]] = None,
) -> np.ndarray:
    """Geometry-preserving foveation: sharp fovea, blurred periphery, no warp.

    A disc around `center` whose area is ~`keep_ratio` of the image stays
    bit-identical to the input; outside it, detail falls off by blending
    progressively stronger Gaussian blurs with radial weights. Every output
    pixel keeps its input coordinates, so intrinsics-based back-projection
    (SpatialVLA's Ego3D pathway) and monocular depth see consistent geometry.

    keep_ratio semantics match the log-polar variant: the fully-sharp area is
    ~keep_ratio of the frame. keep_ratio >= 1 returns the input unchanged;
    keep_ratio <= 0 returns the strongest blur everywhere (information
    removed, but geometry intact -- unlike log-polar's black frame).
    """
    frame = np.asarray(image, dtype=np.uint8)
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(f"Expected HxWx3 image, got {frame.shape}")

    keep_ratio = float(keep_ratio)
    height, width = frame.shape[:2]
    if keep_ratio >= 1.0:
        return frame.copy()

    if center is None:
        center = (width / 2.0, height / 2.0)
    else:
        center = (float(np.clip(center[0], 0, width - 1)),
                  float(np.clip(center[1], 0, height - 1)))

    # Sharp-disc radius from the keep area; blur ramps from r0 to the corner.
    r0 = math.sqrt(max(keep_ratio, 0.0) * height * width / math.pi)
    max_radius = float(
        np.hypot(max(center[0], width - center[0]), max(center[1], height - center[1]))
    )
    ramp = max(max_radius - r0, 1e-6)

    ys, xs = np.mgrid[0:height, 0:width]
    dist = np.hypot(xs - center[0], ys - center[1])
    # 0 inside the fovea, ->1 at the farthest corner.
    t = np.clip((dist - r0) / ramp, 0.0, 1.0).astype(np.float32)

    # Three blur strengths; per-pixel blend picks an effective level ~ t.
    blur_mid = cv2.GaussianBlur(frame, (0, 0), sigmaX=3.0)
    blur_far = cv2.GaussianBlur(frame, (0, 0), sigmaX=9.0)

    w_far = np.clip(2.0 * t - 1.0, 0.0, 1.0)[..., None]      # active t in [0.5, 1]
    w_mid = (np.clip(2.0 * t, 0.0, 1.0)[..., None] - w_far)  # active t in [0, 0.5]
    w_sharp = 1.0 - w_mid - w_far

    out = (frame.astype(np.float32) * w_sharp
           + blur_mid.astype(np.float32) * w_mid
           + blur_far.astype(np.float32) * w_far)
    out = np.clip(np.rint(out), 0, 255).astype(np.uint8)
    # Keep the fovea bit-identical (blending is a no-op there, but rounding
    # of float weights could flip a value; enforce exactly).
    fovea = dist <= r0
    out[fovea] = frame[fovea]
    return out


class MotionGaze:
    """Fovea-center tracker from plain frame differencing.

    update(frame) returns an EMA-smoothed (cx, cy) of where the scene changed
    since the previous frame -- in manipulation that is the moving
    gripper/object locus. Universal plug-in signal: needs only the frames the
    policy already receives (no env oracle, no detector, no model access).
    Falls back to holding the previous center (initially the image center)
    when motion energy is negligible, e.g. on the first frame or a static
    scene.
    """

    def __init__(self, alpha: float = 0.5, noise_floor: float = 15.0,
                 min_pixels: int = 25):
        self.alpha = float(alpha)
        # Per-pixel |diff| below this (0-255 scale) is treated as sensor
        # noise. The gate is a count of super-threshold pixels, NOT a global
        # mean: a moving gripper lights up only a small patch of the frame,
        # which a whole-image mean would wash out and miss.
        self.noise_floor = float(noise_floor)
        self.min_pixels = int(min_pixels)
        self._prev_gray: Optional[np.ndarray] = None
        self._center: Optional[Tuple[float, float]] = None

    def reset(self) -> None:
        self._prev_gray = None
        self._center = None

    def update(self, frame: np.ndarray) -> Tuple[float, float]:
        frame = np.asarray(frame, dtype=np.uint8)
        height, width = frame.shape[:2]
        if self._center is None:
            self._center = (width / 2.0, height / 2.0)

        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY).astype(np.float32)
        prev, self._prev_gray = self._prev_gray, gray
        if prev is None or prev.shape != gray.shape:
            return self._center

        diff = np.abs(gray - prev)
        mask = diff >= self.noise_floor
        if int(mask.sum()) < self.min_pixels:
            return self._center  # static scene / sensor noise: hold gaze

        ys, xs = np.nonzero(mask)
        weights = diff[ys, xs]
        cx = float(np.average(xs, weights=weights))
        cy = float(np.average(ys, weights=weights))

        px, py = self._center
        self._center = (px + self.alpha * (cx - px), py + self.alpha * (cy - py))
        return self._center

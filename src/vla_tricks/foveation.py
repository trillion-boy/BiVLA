"""Geometry-preserving foveation retained as a robustness control.

The transform returns the same spatial resolution, so it does not reduce the
number of OpenVLA visual tokens or model FLOPs.
"""

from __future__ import annotations

import math

import cv2
import numpy as np


def foveate_blur(
    image: np.ndarray,
    keep_ratio: float = 0.20,
    center: tuple[float, float] | None = None,
) -> np.ndarray:
    """Keep a sharp disc and progressively blur the periphery."""
    frame = np.asarray(image)
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(f"expected HxWx3 RGB image, got {frame.shape}")
    if frame.dtype != np.uint8:
        raise TypeError(f"expected uint8 RGB image, got {frame.dtype}")

    keep_ratio = float(np.clip(keep_ratio, 0.0, 1.0))
    if keep_ratio >= 1.0:
        return frame.copy()

    height, width = frame.shape[:2]
    if center is None:
        center = (width / 2.0, height / 2.0)
    cx = float(np.clip(center[0], 0, width - 1))
    cy = float(np.clip(center[1], 0, height - 1))

    sharp_radius = math.sqrt(keep_ratio * height * width / math.pi)
    max_radius = math.hypot(max(cx, width - cx), max(cy, height - cy))
    ramp = max(max_radius - sharp_radius, 1e-6)

    ys, xs = np.mgrid[0:height, 0:width]
    distance = np.hypot(xs - cx, ys - cy)
    t = np.clip((distance - sharp_radius) / ramp, 0.0, 1.0).astype(np.float32)

    middle = cv2.GaussianBlur(frame, (0, 0), sigmaX=3.0)
    far = cv2.GaussianBlur(frame, (0, 0), sigmaX=9.0)
    far_weight = np.clip(2.0 * t - 1.0, 0.0, 1.0)[..., None]
    middle_weight = np.clip(2.0 * t, 0.0, 1.0)[..., None] - far_weight
    sharp_weight = 1.0 - middle_weight - far_weight

    output = (
        frame.astype(np.float32) * sharp_weight
        + middle.astype(np.float32) * middle_weight
        + far.astype(np.float32) * far_weight
    )
    output = np.clip(np.rint(output), 0, 255).astype(np.uint8)
    output[distance <= sharp_radius] = frame[distance <= sharp_radius]
    return output


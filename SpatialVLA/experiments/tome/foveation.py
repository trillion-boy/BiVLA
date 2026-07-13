"""
Log-polar foveation for SpatialVLA observations.

Direct port of the mentor's foveated-input pathway from the RetinaBased
OpenVLA evaluation (RetinaBased/PythonProject/openvla_inference.py:
_uniform_sample_grid + foveate_image_logpolar). The transform is kept
bit-identical so results are comparable across backbones:

  1. warp the RGB frame into log-polar space (cv2.WARP_POLAR_LOG),
  2. subsample a uniform grid whose area is ~keep_ratio of the original
     (rows and cols each scaled by sqrt(keep_ratio)),
  3. resize back to full log-polar resolution,
  4. inverse-warp to Cartesian image space.

Because the subsampling happens in log-polar space, detail is preserved
near the image center and progressively blurred toward the periphery --
the model still receives a normal HxWx3 image, only its content changes.
"""
from __future__ import annotations

import math
from typing import Tuple

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


def foveate_image_logpolar(image: np.ndarray, keep_ratio: float) -> np.ndarray:
    frame = np.asarray(image, dtype=np.uint8)
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(f"Expected HxWx3 image, got {frame.shape}")

    if keep_ratio <= 0.0:
        return np.zeros_like(frame)

    height, width = frame.shape[:2]
    center = (width / 2.0, height / 2.0)
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

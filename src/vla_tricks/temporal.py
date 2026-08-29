"""Temporal controls: the fixed baseline and a conservative replacement."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np


def apply_action_repeat(actions: np.ndarray, repeat: int = 2) -> np.ndarray:
    """Repeat each action consecutively; retained as a negative control."""
    array = np.atleast_2d(np.asarray(actions))
    if repeat < 1:
        raise ValueError("repeat must be at least one")
    return np.repeat(array, int(repeat), axis=0)


def _frame_signature(image: np.ndarray, stride: int) -> np.ndarray:
    frame = np.asarray(image)
    if frame.ndim != 3 or frame.shape[-1] != 3:
        raise ValueError(f"expected HxWx3 image, got {frame.shape}")
    # Sampling is deliberately cheap. Float32 avoids uint8 subtraction wrap.
    return frame[::stride, ::stride].astype(np.float32) / 255.0


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    av = np.asarray(a, dtype=np.float32).reshape(-1)
    bv = np.asarray(b, dtype=np.float32).reshape(-1)
    denominator = float(np.linalg.norm(av) * np.linalg.norm(bv))
    if denominator <= 1e-12:
        return 1.0 if np.allclose(av, bv) else 0.0
    return float(np.dot(av, bv) / denominator)


def _maximum_local_mae(
    previous: np.ndarray, current: np.ndarray, grid_size: int
) -> float:
    """Detect localized changes that a whole-frame average can hide."""
    if grid_size < 1:
        raise ValueError("local_grid_size must be positive")
    if previous.shape != current.shape:
        raise ValueError(f"frame signatures differ: {previous.shape} and {current.shape}")
    difference = np.abs(current - previous)
    height, width = difference.shape[:2]
    if height % grid_size == 0 and width % grid_size == 0:
        patch_height, patch_width = height // grid_size, width // grid_size
        patches = difference.reshape(
            grid_size, patch_height, grid_size, patch_width, *difference.shape[2:]
        ).transpose(0, 2, 1, 3, *range(4, difference.ndim + 2))
        return float(patches.mean(axis=tuple(range(2, patches.ndim))).max())
    row_groups = np.array_split(np.arange(difference.shape[0]), grid_size)
    column_groups = np.array_split(np.arange(difference.shape[1]), grid_size)
    return max(
        float(difference[np.ix_(rows, columns)].mean())
        for rows in row_groups
        for columns in column_groups
        if len(rows) and len(columns)
    )


@dataclass
class ConservativeActionReuse:
    """Skip inference only when scene and recent policy actions are stable.

    Unlike unconditional action repeat, this controller observes the current
    image before deciding. It also disables reuse near low-motion/contact
    phases and across gripper transitions. Thresholds are action-normalization
    dependent and must be selected on a held-out calibration split.
    """

    max_frame_mae: float = 0.01
    max_local_patch_mae: float | None = 0.03
    local_grid_size: int = 8
    min_action_cosine: float = 0.995
    min_translation_norm: float = 0.01
    max_consecutive_reuse: int = 1
    signature_stride: int = 8
    translation_dims: tuple[int, ...] = (0, 1, 2)
    gripper_dim: int = 6
    _previous_signature: np.ndarray | None = field(default=None, init=False)
    _inferred_actions: list[np.ndarray] = field(default_factory=list, init=False)
    _last_action: np.ndarray | None = field(default=None, init=False)
    _consecutive_reuse: int = field(default=0, init=False)
    calls: int = field(default=0, init=False)
    reuses: int = field(default=0, init=False)

    def reset(self) -> None:
        self._previous_signature = None
        self._inferred_actions.clear()
        self._last_action = None
        self._consecutive_reuse = 0
        self.calls = 0
        self.reuses = 0

    def _can_reuse(self, signature: np.ndarray) -> bool:
        if (
            self._previous_signature is None
            or self._last_action is None
            or len(self._inferred_actions) < 2
            or self._consecutive_reuse >= self.max_consecutive_reuse
        ):
            return False
        frame_mae = float(np.mean(np.abs(signature - self._previous_signature)))
        if frame_mae > self.max_frame_mae:
            return False
        if self.max_local_patch_mae is not None:
            local_mae = _maximum_local_mae(
                self._previous_signature, signature, self.local_grid_size
            )
            if local_mae > self.max_local_patch_mae:
                return False

        previous, current = self._inferred_actions[-2:]
        if _cosine(previous[:6], current[:6]) < self.min_action_cosine:
            return False
        translation = np.asarray(current)[list(self.translation_dims)]
        if float(np.linalg.norm(translation)) < self.min_translation_norm:
            return False
        if np.sign(previous[self.gripper_dim]) != np.sign(current[self.gripper_dim]):
            return False
        return True

    def step(self, image: np.ndarray, infer: Callable[[], np.ndarray]) -> tuple[np.ndarray, bool]:
        """Return `(action, reused)` and call `infer` only when required."""
        signature = _frame_signature(image, self.signature_stride)
        if self._can_reuse(signature):
            self._consecutive_reuse += 1
            self.reuses += 1
            self._previous_signature = signature
            return self._last_action.copy(), True

        action = np.asarray(infer(), dtype=np.float32).reshape(-1)
        if action.size <= self.gripper_dim:
            raise ValueError("action does not contain the configured gripper dimension")
        self.calls += 1
        self._consecutive_reuse = 0
        self._previous_signature = signature
        self._last_action = action.copy()
        self._inferred_actions.append(action.copy())
        self._inferred_actions = self._inferred_actions[-2:]
        return action, False

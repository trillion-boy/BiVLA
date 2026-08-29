"""Interaction-aware temporal perception candidates for OpenVLA.

The selector combines three complementary signals before declaring a visual
patch reusable: local motion, intrinsic image entropy, and optional task
relevance.  The same decision can drive either projected-token fusion (a
temporal-denoising candidate) or a model-internal cache implementation (an
acceleration candidate).  Neither role is an accuracy result without paired
robot rollouts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import cv2
import numpy as np
import torch


def _validate_rgb(image: np.ndarray) -> np.ndarray:
    frame = np.asarray(image)
    if frame.ndim != 3 or frame.shape[-1] != 3:
        raise ValueError(f"expected HxWx3 image, got {frame.shape}")
    if frame.dtype != np.uint8:
        raise TypeError(f"expected uint8 RGB image, got {frame.dtype}")
    return frame


def _patch_view(values: np.ndarray, grid_size: tuple[int, int]) -> np.ndarray:
    rows, columns = map(int, grid_size)
    if rows < 1 or columns < 1:
        raise ValueError("grid dimensions must be positive")
    height, width = values.shape[:2]
    if height % rows or width % columns:
        raise ValueError(
            f"image shape {(height, width)} is not divisible by grid {grid_size}"
        )
    patch_height, patch_width = height // rows, width // columns
    remaining = values.shape[2:]
    reshaped = values.reshape(rows, patch_height, columns, patch_width, *remaining)
    axes = (0, 2, 1, 3, *range(4, reshaped.ndim))
    return reshaped.transpose(axes).reshape(rows * columns, patch_height, patch_width, *remaining)


def patch_motion(
    previous_image: np.ndarray,
    current_image: np.ndarray,
    *,
    grid_size: tuple[int, int] = (16, 16),
) -> np.ndarray:
    """Return normalized RGB mean-absolute-change for each spatial patch."""
    previous = _validate_rgb(previous_image)
    current = _validate_rgb(current_image)
    if previous.shape != current.shape:
        raise ValueError(f"frame shapes differ: {previous.shape} and {current.shape}")
    difference = np.abs(current.astype(np.float32) - previous.astype(np.float32)) / 255.0
    return _patch_view(difference, grid_size).mean(axis=(1, 2, 3))


def patch_entropy(
    image: np.ndarray,
    *,
    grid_size: tuple[int, int] = (16, 16),
    bins: int = 16,
) -> np.ndarray:
    """Return normalized grayscale histogram entropy for each spatial patch."""
    frame = _validate_rgb(image)
    if bins < 2 or bins > 256:
        raise ValueError("bins must be between 2 and 256")
    grayscale = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    patches = _patch_view(grayscale, grid_size)
    quantized = np.minimum((patches.astype(np.uint16) * bins) // 256, bins - 1)
    offsets = np.arange(len(patches), dtype=np.int64)[:, None] * bins
    flattened = quantized.reshape(len(patches), -1).astype(np.int64) + offsets
    counts = np.bincount(flattened.ravel(), minlength=len(patches) * bins).reshape(
        len(patches), bins
    )
    probabilities = counts.astype(np.float64) / flattened.shape[1]
    logarithms = np.zeros_like(probabilities)
    np.log2(probabilities, out=logarithms, where=probabilities > 0)
    return (-(probabilities * logarithms).sum(axis=1) / np.log2(bins)).astype(
        np.float32
    )


def _top_fraction(values: np.ndarray, fraction: float) -> np.ndarray:
    fraction = float(np.clip(fraction, 0.0, 1.0))
    count = int(np.ceil(fraction * len(values)))
    selected = np.zeros(len(values), dtype=bool)
    if count:
        indices = np.argpartition(values, len(values) - count)[-count:]
        selected[indices] = True
    return selected


def _dilate(mask: np.ndarray, grid_size: tuple[int, int], radius: int) -> np.ndarray:
    if radius < 0:
        raise ValueError("protect_radius must be non-negative")
    if radius == 0:
        return mask.copy()
    grid = mask.reshape(grid_size).astype(np.uint8)
    width = 2 * radius + 1
    kernel = np.ones((width, width), dtype=np.uint8)
    return cv2.dilate(grid, kernel, iterations=1).astype(bool).reshape(-1)


@dataclass(frozen=True)
class PatchReuseDecision:
    """Auditable patch-level decision shared by fusion and cache methods."""

    reusable_patch_ids: tuple[int, ...]
    protected_patch_ids: tuple[int, ...]
    motion: np.ndarray
    entropy: np.ndarray
    task_relevance: np.ndarray

    @property
    def reusable_fraction(self) -> float:
        return len(self.reusable_patch_ids) / len(self.motion)


def select_reusable_patches(
    previous_image: np.ndarray,
    current_image: np.ndarray,
    *,
    task_relevance: Sequence[float] | np.ndarray | None = None,
    always_protect: Sequence[int] = (),
    grid_size: tuple[int, int] = (16, 16),
    motion_threshold: float = 0.01,
    entropy_protect_fraction: float = 0.15,
    task_protect_fraction: float = 0.20,
    protect_radius: int = 1,
    max_reuse_fraction: float = 0.50,
) -> PatchReuseDecision:
    """Select patches that are static, low-information, and task-irrelevant.

    High-motion, high-entropy, task-relevant, explicitly protected, and their
    neighboring patches are always recomputed.  When more safe candidates are
    available than the reuse budget permits, the lowest combined-risk patches
    are selected deterministically.
    """
    if motion_threshold < 0:
        raise ValueError("motion_threshold must be non-negative")
    if not 0.0 <= max_reuse_fraction <= 1.0:
        raise ValueError("max_reuse_fraction must be in [0, 1]")

    motion = patch_motion(previous_image, current_image, grid_size=grid_size)
    entropy = patch_entropy(current_image, grid_size=grid_size)
    patch_count = len(motion)
    if task_relevance is None:
        relevance = np.zeros(patch_count, dtype=np.float32)
    else:
        relevance = np.asarray(task_relevance, dtype=np.float32).reshape(-1)
        if len(relevance) != patch_count:
            raise ValueError(
                f"task relevance has {len(relevance)} entries; expected {patch_count}"
            )
        minimum, maximum = float(relevance.min()), float(relevance.max())
        relevance = (
            (relevance - minimum) / (maximum - minimum)
            if maximum > minimum
            else np.zeros_like(relevance)
        )

    protected = motion > motion_threshold
    protected |= _top_fraction(entropy, entropy_protect_fraction)
    if task_relevance is not None:
        protected |= _top_fraction(relevance, task_protect_fraction)
    for index in always_protect:
        index = int(index)
        if index < 0 or index >= patch_count:
            raise IndexError(f"protected patch {index} is outside [0, {patch_count})")
        protected[index] = True
    protected = _dilate(protected, grid_size, protect_radius)

    candidates = np.flatnonzero(~protected)
    budget = min(len(candidates), int(np.floor(max_reuse_fraction * patch_count)))
    if budget:
        risk = motion + entropy + relevance
        order = np.lexsort((candidates, risk[candidates]))
        reusable = tuple(sorted(map(int, candidates[order[:budget]])))
    else:
        reusable = ()
    return PatchReuseDecision(
        reusable_patch_ids=reusable,
        protected_patch_ids=tuple(map(int, np.flatnonzero(protected))),
        motion=motion.copy(),
        entropy=entropy.copy(),
        task_relevance=relevance.copy(),
    )


@torch.no_grad()
def fuse_projected_tokens(
    current_tokens: torch.Tensor,
    previous_tokens: torch.Tensor,
    reusable_patch_ids: Sequence[int],
) -> torch.Tensor:
    """Replace selected current projected tokens with historical tokens."""
    if current_tokens.ndim != 3 or current_tokens.shape[0] != 1:
        raise ValueError("expected current tokens shaped [1, patches, hidden]")
    if current_tokens.shape != previous_tokens.shape:
        raise ValueError(
            f"token shapes differ: {tuple(current_tokens.shape)} and {tuple(previous_tokens.shape)}"
        )
    indices = sorted(set(map(int, reusable_patch_ids)))
    if any(index < 0 or index >= current_tokens.shape[1] for index in indices):
        raise IndexError("reusable patch index is outside the token sequence")
    fused = current_tokens.clone()
    if indices:
        index_tensor = torch.tensor(indices, device=fused.device, dtype=torch.long)
        fused[:, index_tensor] = previous_tokens.to(fused.device)[:, index_tensor]
    return fused


class InteractionAwareTemporalFusion:
    """Projector hook for training-free temporal token fusion.

    Call :meth:`prepare` with the raw RGB observation immediately before each
    OpenVLA inference.  Attach the controller to ``model.projector``.  Dense
    keyframes prevent indefinite propagation of historical representations.
    This controller does not itself reduce FLOPs; its decision is also exposed
    so the same mask can drive a selective cache implementation.
    """

    def __init__(
        self,
        *,
        keyframe_interval: int = 3,
        grid_size: tuple[int, int] = (16, 16),
        motion_threshold: float = 0.01,
        entropy_protect_fraction: float = 0.15,
        task_protect_fraction: float = 0.20,
        protect_radius: int = 1,
        max_reuse_fraction: float = 0.50,
    ) -> None:
        if keyframe_interval < 1:
            raise ValueError("keyframe_interval must be positive")
        self.keyframe_interval = int(keyframe_interval)
        self.selector_kwargs = {
            "grid_size": grid_size,
            "motion_threshold": motion_threshold,
            "entropy_protect_fraction": entropy_protect_fraction,
            "task_protect_fraction": task_protect_fraction,
            "protect_radius": protect_radius,
            "max_reuse_fraction": max_reuse_fraction,
        }
        self._handle: Any | None = None
        self._previous_image: np.ndarray | None = None
        self._previous_tokens: torch.Tensor | None = None
        self._pending_image: np.ndarray | None = None
        self._decision: PatchReuseDecision | None = None
        self.steps = 0
        self.fused_steps = 0

    @property
    def decision(self) -> PatchReuseDecision | None:
        return self._decision

    def attach(self, projector: torch.nn.Module) -> None:
        if self._handle is not None:
            raise RuntimeError("temporal fusion is already attached")
        self._handle = projector.register_forward_hook(self._hook)

    def detach(self) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    def reset(self) -> None:
        self._previous_image = None
        self._previous_tokens = None
        self._pending_image = None
        self._decision = None
        self.steps = 0
        self.fused_steps = 0

    def prepare(
        self,
        image: np.ndarray,
        *,
        task_relevance: Sequence[float] | np.ndarray | None = None,
        always_protect: Sequence[int] = (),
        force_keyframe: bool = False,
    ) -> PatchReuseDecision | None:
        frame = _validate_rgb(image).copy()
        keyframe = force_keyframe or self.steps % self.keyframe_interval == 0
        if self._previous_image is None or keyframe:
            self._decision = None
        else:
            self._decision = select_reusable_patches(
                self._previous_image,
                frame,
                task_relevance=task_relevance,
                always_protect=always_protect,
                **self.selector_kwargs,
            )
        self._pending_image = frame
        self.steps += 1
        return self._decision

    def _hook(self, module: torch.nn.Module, args: tuple[Any, ...], output: Any) -> Any:
        if not torch.is_tensor(output):
            raise TypeError("OpenVLA projector hook expected a tensor output")
        if self._pending_image is None:
            raise RuntimeError("call temporal fusion prepare() before model inference")
        fused = output
        if self._decision is not None and self._previous_tokens is not None:
            fused = fuse_projected_tokens(
                output, self._previous_tokens, self._decision.reusable_patch_ids
            )
            if self._decision.reusable_patch_ids:
                self.fused_steps += 1
        self._previous_tokens = fused.detach().clone()
        self._previous_image = self._pending_image
        self._pending_image = None
        return fused

    def __enter__(self) -> "InteractionAwareTemporalFusion":
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.detach()

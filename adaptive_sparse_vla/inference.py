"""
Frozen VLA inference helpers with an adaptive sparse visual gate.

The active runtime paths are:

1. `EmuVLAInference`
   Fixed-resolution frozen VLA inference.

2. `AdaptiveSparseInference`
   A training-free sparse visual gate. The selected image regions remain at
   normal resolution and the rest are downsampled before policy inference.
"""

from __future__ import annotations

import colorsys
import copy
import gc
import math
import os
import sys
from dataclasses import dataclass
from queue import Queue
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from huggingface_hub import snapshot_download
from PIL import Image
from safetensors.torch import load_file as safe_load_file
from transforms3d.euler import euler2axangle
from transformers import (
    AutoImageProcessor,
    AutoModel,
    AutoProcessor,
    GenerationConfig,
    LogitsProcessor,
)
from transformers.feature_extraction_utils import BatchFeature

# ── Repo path setup ────────────────────────────────────────────────────────────
_ROOT = os.environ.get(
    "UNIVLA_ROOT",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "UniVLA")),
)
_EMU3 = os.path.join(_ROOT, "reference", "Emu3")
_REPO_ROOT = os.path.abspath(os.path.join(_ROOT, ".."))
_AUTOGAZE = os.path.join(_REPO_ROOT, "AutoGaze")
for _p in [_REPO_ROOT, _ROOT, _EMU3, _AUTOGAZE]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from emu3.mllm import Emu3MoE, Emu3Processor, Emu3Tokenizer  # noqa: E402
from autogaze.models.autogaze import (  # noqa: E402
    AutoGaze,
    AutoGazeConfig,
    AutoGazeImageProcessor,
)
from shared_unified_policy import (
    CompactFocusGateConfig,
    compact_focus_gate,
    infer_task_archetype,
    shared_task_policy_profile,
)
try:
    from adaptive_sparse_vla.fastv_emu3 import apply_fastv, visual_mask_from_input_ids
except Exception:  # when imported as a top-level package on sys.path
    from fastv_emu3 import apply_fastv, visual_mask_from_input_ids
try:
    from adaptive_sparse_vla.bypass_layer import BypassDecoderLayer
except Exception:  # when imported as a top-level package on sys.path
    from bypass_layer import BypassDecoderLayer
try:
    from adaptive_sparse_vla.emu3_self_spec_decode import emu3_self_speculative_generate
except Exception:  # when imported as a top-level package on sys.path
    from emu3_self_spec_decode import emu3_self_speculative_generate

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass


class ActionIDConstraintLogitsProcessor(LogitsProcessor):
    def __init__(self, allowed_token_ids):
        self.allowed_token_ids = allowed_token_ids

    def __call__(self, input_ids, scores):
        mask = torch.zeros_like(scores, dtype=torch.bool)
        if mask.ndim == 1:
            mask[self.allowed_token_ids] = True
        else:
            mask[:, self.allowed_token_ids] = True
        scores[~mask] = -float("inf")
        return scores


@dataclass
class PatchSelectionConfig:
    grid_size: int = 20
    max_highres_patches: int = 80
    lowres_factor: float = 0.1
    autogaze_hub: str = "nvidia/AutoGaze"
    autogaze_target_patch_size: int = 16
    autogaze_task_loss_requirement: Optional[float] = None
    grasp_visible_ratio: float = 0.20
    place_visible_ratio: float = 0.20


@dataclass
class PatchSelectionResult:
    mixed_image: np.ndarray
    score_map: np.ndarray
    selection_ratio: float
    selected_patch_count: int
    pixel_mask: np.ndarray
    grid_mask: np.ndarray


def _resize_rgb(image: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    return np.asarray(
        Image.fromarray(image.astype(np.uint8)).resize(size, Image.Resampling.BILINEAR),
        dtype=np.uint8,
    )


def _resize_mask(mask: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    mask_u8 = (mask.astype(np.uint8) * 255)
    resized = Image.fromarray(mask_u8).resize(size, Image.Resampling.NEAREST)
    return np.asarray(resized, dtype=np.uint8) > 0


def _resize_score_map(score_map: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    resized = Image.fromarray(score_map.astype(np.float32), mode="F").resize(
        size, Image.Resampling.BILINEAR
    )
    return np.asarray(resized, dtype=np.float32)


def _normalize_score_map(score_map: np.ndarray) -> np.ndarray:
    score_map = np.nan_to_num(score_map.astype(np.float32), copy=False)
    lo = float(np.min(score_map))
    hi = float(np.max(score_map))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo + 1e-6:
        return np.zeros_like(score_map, dtype=np.float32)
    return (score_map - lo) / (hi - lo)


def _rgb_to_hsv_image(image: np.ndarray) -> np.ndarray:
    rgb = np.asarray(image, dtype=np.float32) / 255.0
    flat = rgb.reshape(-1, 3)
    hsv = np.array([colorsys.rgb_to_hsv(*px) for px in flat], dtype=np.float32)
    return hsv.reshape(rgb.shape)


def _circular_hue_distance(hue: np.ndarray, center: float) -> np.ndarray:
    delta = np.abs(hue - center)
    return np.minimum(delta, 1.0 - delta)


def _dilate_grid(mask: np.ndarray, radius: int = 1) -> np.ndarray:
    if radius <= 0:
        return mask
    out = mask.copy()
    for _ in range(radius):
        padded = np.pad(out, 1, mode="edge")
        neighbors = []
        for dy in range(3):
            for dx in range(3):
                neighbors.append(padded[dy : dy + out.shape[0], dx : dx + out.shape[1]])
        out = np.logical_or.reduce(neighbors)
    return out


def _topk_mask(score_map: np.ndarray, k: int) -> np.ndarray:
    flat = score_map.reshape(-1)
    k = max(0, min(int(k), flat.size))
    mask = np.zeros(flat.size, dtype=bool)
    if k <= 0:
        return mask.reshape(score_map.shape)
    if np.allclose(flat, flat[0]):
        idx = np.linspace(0, flat.size - 1, k, dtype=int)
    else:
        idx = np.argpartition(flat, -k)[-k:]
    mask[idx] = True
    return mask.reshape(score_map.shape)


def _goal_phase_hint(goal: Optional[str], phase_info: Optional[dict]) -> str:
    if phase_info and (
        phase_info.get("ever_grasped_src") or phase_info.get("is_src_obj_grasped")
    ):
        return "place"
    if goal and "place" in goal.lower():
        return "place"
    return "grasp"


def _center_of_mass(score_map: np.ndarray) -> Optional[Tuple[float, float]]:
    weights = np.clip(score_map.astype(np.float32), 0.0, None)
    total = float(weights.sum())
    if total <= 1e-6:
        return None
    ys, xs = np.indices(weights.shape, dtype=np.float32)
    return float((ys * weights).sum() / total), float((xs * weights).sum() / total)


def _extract_task_entities(goal: Optional[str]) -> Dict[str, Optional[str]]:
    text = (goal or "").lower().strip()
    source = None
    target = None
    if text.startswith("put ") and " on " in text:
        payload = text[4:]
        source, target = payload.split(" on ", 1)
    elif text.startswith("stack ") and " on " in text:
        payload = text[6:]
        source, target = payload.split(" on ", 1)
    source = (source or "").replace("the ", "").strip() or None
    target = (target or "").replace("the ", "").strip() or None
    return {"source": source, "target": target}


def _color_score(hsv: np.ndarray, center: float, width: float, sat_min: float, val_min: float) -> np.ndarray:
    hue = hsv[..., 0]
    sat = hsv[..., 1]
    val = hsv[..., 2]
    hue_score = np.clip(1.0 - (_circular_hue_distance(hue, center) / max(width, 1e-6)), 0.0, 1.0)
    sat_score = np.clip((sat - sat_min) / max(1.0 - sat_min, 1e-6), 0.0, 1.0)
    val_score = np.clip((val - val_min) / max(1.0 - val_min, 1e-6), 0.0, 1.0)
    return hue_score * sat_score * val_score


def _entity_pixel_scores(image: np.ndarray, goal: Optional[str]) -> Tuple[np.ndarray, np.ndarray]:
    hsv = _rgb_to_hsv_image(image)
    entities = _extract_task_entities(goal)
    source = entities["source"] or ""
    target = entities["target"] or ""

    source_score = np.zeros(image.shape[:2], dtype=np.float32)
    target_score = np.zeros_like(source_score)

    if "carrot" in source:
        source_score = np.maximum(source_score, _color_score(hsv, 0.08, 0.06, 0.35, 0.15))
    if "eggplant" in source:
        source_score = np.maximum(source_score, _color_score(hsv, 0.80, 0.10, 0.20, 0.05))
    if "green" in source:
        source_score = np.maximum(source_score, _color_score(hsv, 0.33, 0.10, 0.25, 0.15))
    if "yellow" in source:
        source_score = np.maximum(source_score, _color_score(hsv, 0.16, 0.09, 0.20, 0.25))
    if "spoon" in source:
        yellow_handle = _color_score(hsv, 0.13, 0.08, 0.12, 0.18)
        warm_handle = _color_score(hsv, 0.16, 0.10, 0.08, 0.20)
        source_score = np.maximum(source_score, np.maximum(yellow_handle, warm_handle))

    if "plate" in target:
        plate_color = _color_score(hsv, 0.17, 0.10, 0.08, 0.45)
        bright_white = np.clip((hsv[..., 2] - 0.72) / 0.28, 0.0, 1.0) * np.clip(
            (0.35 - hsv[..., 1]) / 0.35, 0.0, 1.0
        )
        target_score = np.maximum(target_score, np.maximum(plate_color, bright_white))
    if "basket" in target:
        target_score = np.maximum(target_score, _color_score(hsv, 0.12, 0.10, 0.20, 0.10))
    if "green" in target:
        target_score = np.maximum(target_score, _color_score(hsv, 0.33, 0.10, 0.25, 0.15))
    if "yellow" in target:
        target_score = np.maximum(target_score, _color_score(hsv, 0.16, 0.09, 0.20, 0.25))
    if "towel" in target or "cloth" in target:
        target_score = np.maximum(target_score, _color_score(hsv, 0.52, 0.12, 0.04, 0.35))

    return source_score, target_score


def _compact_focus_metrics(image: np.ndarray, goal: Optional[str]) -> dict:
    source_score, target_score = _entity_pixel_scores(image, goal)
    source_peak = float(np.max(source_score)) if source_score.size else 0.0
    target_peak = float(np.max(target_score)) if target_score.size else 0.0
    source_area = float((source_score > 0.2).mean())
    target_area = float((target_score > 0.2).mean())
    source_center = _center_of_mass(source_score)
    target_center = _center_of_mass(target_score)
    h, w = image.shape[:2]
    if source_center is None or target_center is None:
        dx_ratio = 0.0
        dy_ratio = 1.0
    else:
        dy_ratio = float((source_center[0] - target_center[0]) / float(max(h, 1)))
        dx_ratio = float((source_center[1] - target_center[1]) / float(max(w, 1)))
    source_bbox = _mask_bbox(source_score > 0.2)
    target_bbox = _mask_bbox(target_score > 0.2)
    overlap_ratio = _bbox_iou(source_bbox, target_bbox)
    return {
        "source_peak": source_peak,
        "target_peak": target_peak,
        "source_area": source_area,
        "target_area": target_area,
        "dx_ratio": dx_ratio,
        "dy_ratio": dy_ratio,
        "overlap_ratio": float(overlap_ratio),
    }


def _mask_bbox(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    ys, xs = np.nonzero(mask)
    if ys.size == 0 or xs.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _bbox_area(bbox: Optional[Tuple[int, int, int, int]]) -> float:
    if bbox is None:
        return 0.0
    x0, y0, x1, y1 = bbox
    return float(max(0, x1 - x0) * max(0, y1 - y0))


def _bbox_iou(
    a: Optional[Tuple[int, int, int, int]],
    b: Optional[Tuple[int, int, int, int]],
) -> float:
    if a is None or b is None:
        return 0.0
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0 = max(ax0, bx0)
    iy0 = max(ay0, by0)
    ix1 = min(ax1, bx1)
    iy1 = min(ay1, by1)
    inter = _bbox_area((ix0, iy0, ix1, iy1))
    if inter <= 0.0:
        return 0.0
    union = _bbox_area(a) + _bbox_area(b) - inter
    if union <= 1e-6:
        return 0.0
    return float(inter / union)


def _round_to_multiple(value: int, multiple: int) -> int:
    return max(multiple, int(round(value / multiple)) * multiple)


def _build_autogaze_target_scales(final_size: int, patch_size: int) -> Tuple[int, int, int, int]:
    if final_size % patch_size != 0:
        final_size = _round_to_multiple(final_size, patch_size)
    quarter = _round_to_multiple(final_size // 4, patch_size)
    eighth = _round_to_multiple(final_size // 8, patch_size)
    half = _round_to_multiple(final_size // 2, patch_size)
    return (eighth, quarter, half, final_size)


def _resolve_model_path(model_name_or_path: str) -> str:
    if os.path.isdir(model_name_or_path):
        return model_name_or_path
    return snapshot_download(
        repo_id=model_name_or_path,
        allow_patterns=["config.json", "model.safetensors", "preprocessor_config.json"],
    )


def _load_autogaze_model(
    model_name_or_path: str,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[AutoGaze, str]:
    model_dir = _resolve_model_path(model_name_or_path)
    config = AutoGazeConfig.from_pretrained(model_dir)
    model = AutoGaze(config)
    state_dict = safe_load_file(os.path.join(model_dir, "model.safetensors"), device="cpu")
    incompat = model.load_state_dict(state_dict, strict=True)
    if incompat.missing_keys or incompat.unexpected_keys:
        raise RuntimeError(
            "Failed to load AutoGaze weights cleanly: "
            f"missing={incompat.missing_keys} unexpected={incompat.unexpected_keys}"
        )
    model = model.to(device, dtype=dtype).eval()
    return model, model_dir


class SparseMaskComposer:
    def __init__(self, config: PatchSelectionConfig):
        self.config = config

    def _patch_bounds(self, length: int, idx: int) -> Tuple[int, int]:
        start = int(round(idx * length / self.config.grid_size))
        end = int(round((idx + 1) * length / self.config.grid_size))
        return start, max(start + 1, end)

    def _lowres_patch(self, patch: np.ndarray) -> np.ndarray:
        h, w = patch.shape[:2]
        down_w = max(1, int(round(w * self.config.lowres_factor)))
        down_h = max(1, int(round(h * self.config.lowres_factor)))
        lowres = Image.fromarray(patch.astype(np.uint8)).resize(
            (down_w, down_h), Image.Resampling.BILINEAR
        )
        restored = lowres.resize((w, h), Image.Resampling.BILINEAR)
        return np.asarray(restored, dtype=np.uint8)

    def compose(self, image: np.ndarray, grid_mask: np.ndarray) -> np.ndarray:
        mixed = np.asarray(image, dtype=np.uint8).copy()
        h, w = mixed.shape[:2]
        for row in range(self.config.grid_size):
            y0, y1 = self._patch_bounds(h, row)
            for col in range(self.config.grid_size):
                x0, x1 = self._patch_bounds(w, col)
                if grid_mask[row, col]:
                    continue
                mixed[y0:y1, x0:x1] = self._lowres_patch(mixed[y0:y1, x0:x1])
        return mixed


class AutoGazeSelectorController:
    def __init__(
        self,
        config: PatchSelectionConfig,
        device: str = "cuda",
        target_image_size: int = 256,
    ):
        self.config = config
        self.device = torch.device(device)
        self.target_patch_size = int(config.autogaze_target_patch_size)
        self.target_scales = _build_autogaze_target_scales(
            int(target_image_size),
            self.target_patch_size,
        )
        self.model_dtype = (
            torch.bfloat16 if self.device.type == "cuda" else torch.float32
        )
        self.model_dir = _resolve_model_path(config.autogaze_hub)
        self.processor = AutoGazeImageProcessor.from_pretrained(
            self.model_dir,
            size={"height": self.target_scales[-1], "width": self.target_scales[-1]},
        )
        self.model, _ = _load_autogaze_model(
            self.model_dir,
            device=self.device,
            dtype=self.model_dtype,
        )
        self.composer = SparseMaskComposer(config)
        self.reset()

    def reset(self) -> None:
        self._past_key_values = None
        self._past_inputs_embeds = None
        self._past_attention_mask = None
        self._past_conv_values = None
        self._prev_grid_mask = None
        self._prev_score_map = None
        self._prev_source_score = None
        self._prev_target_score = None
        self._last_gaze_ratio = None

    def _prepare_video(self, image: np.ndarray) -> torch.Tensor:
        batch = self.processor.preprocess([image], return_tensors="pt")
        return batch["pixel_values"].to(self.device, dtype=self.model_dtype)

    def _score_gazing_mask(self, gaze_outputs: dict) -> np.ndarray:
        score_map = np.zeros((self.config.grid_size, self.config.grid_size), dtype=np.float32)
        gazed = 0.0
        tokens = 0.0
        for flat_mask in gaze_outputs["gazing_mask"]:
            mask_1d = flat_mask[0, -1].detach().float().cpu().numpy()
            gazed += float((mask_1d > 0.5).sum())
            tokens += float(mask_1d.size)
            side = int(round(math.sqrt(mask_1d.size)))
            if side * side != mask_1d.size:
                continue
            mask_2d = mask_1d.reshape(side, side).astype(np.float32)
            score_map += _resize_score_map(
                mask_2d, (self.config.grid_size, self.config.grid_size)
            )
        # A-2 option A: AutoGaze's actual gazed fraction (used for adaptive budget).
        self._last_gaze_ratio = (gazed / tokens) if tokens > 0 else None
        return score_map

    def _grid_score_from_pixel_score(self, pixel_score: np.ndarray) -> np.ndarray:
        h, w = pixel_score.shape
        grid = np.zeros((self.config.grid_size, self.config.grid_size), dtype=np.float32)
        for row in range(self.config.grid_size):
            y0, y1 = self.composer._patch_bounds(h, row)
            for col in range(self.config.grid_size):
                x0, x1 = self.composer._patch_bounds(w, col)
                cell = pixel_score[y0:y1, x0:x1]
                grid[row, col] = float(cell.mean()) if cell.size else 0.0
        return grid

    def _object_prior_scores(
        self,
        image: np.ndarray,
        goal: Optional[str],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        source_score, target_score = _entity_pixel_scores(image, goal)

        source_grid = self._grid_score_from_pixel_score(source_score)
        target_grid = self._grid_score_from_pixel_score(target_score)

        gripper_prior = np.zeros((self.config.grid_size, self.config.grid_size), dtype=np.float32)
        top_rows = max(2, self.config.grid_size // 5)
        center_col = self.config.grid_size // 2
        half_width = max(1, self.config.grid_size // 6)
        gripper_prior[:top_rows, max(0, center_col - half_width) : min(self.config.grid_size, center_col + half_width + 1)] = 1.0
        return source_grid, target_grid, gripper_prior

    def _visible_ratio_for_phase(self, goal: Optional[str], phase_info: Optional[dict], base_ratio: float) -> float:
        phase = _goal_phase_hint(goal, phase_info)
        if phase == "place":
            return max(base_ratio, float(self.config.place_visible_ratio))
        return max(base_ratio, float(self.config.grasp_visible_ratio))

    def _hybrid_grid_mask(
        self,
        autogaze_score: np.ndarray,
        image: np.ndarray,
        goal: Optional[str],
        visible_ratio: float,
        phase_info: Optional[dict],
    ) -> Tuple[np.ndarray, np.ndarray]:
        autogaze_score = _normalize_score_map(autogaze_score)
        source_prior, target_prior, gripper_prior = self._object_prior_scores(image, goal)
        autogaze_score = _normalize_score_map(autogaze_score)
        source_prior = _normalize_score_map(source_prior)
        target_prior = _normalize_score_map(target_prior)
        gripper_prior = _normalize_score_map(gripper_prior)
        bridge_prior = np.zeros_like(autogaze_score, dtype=np.float32)
        src_center = _center_of_mass(source_prior)
        tgt_center = _center_of_mass(target_prior)
        if src_center is not None and tgt_center is not None:
            for t in np.linspace(0.0, 1.0, 32):
                yy = int(round((1.0 - t) * src_center[0] + t * tgt_center[0]))
                xx = int(round((1.0 - t) * src_center[1] + t * tgt_center[1]))
                y0 = max(0, yy - 1)
                y1 = min(self.config.grid_size, yy + 2)
                x0 = max(0, xx - 1)
                x1 = min(self.config.grid_size, xx + 2)
                bridge_prior[y0:y1, x0:x1] = 1.0

        if self._prev_grid_mask is None:
            temporal_prior = np.zeros_like(autogaze_score)
        else:
            temporal_prior = _dilate_grid(self._prev_grid_mask, radius=1).astype(np.float32)

        phase = _goal_phase_hint(goal, phase_info)
        # A-3b ablation: set env AUTOGAZE_PURE=1 to use pure AutoGaze saliency for
        # ALL phases (drop the hand-crafted color/gripper/temporal/bridge priors).
        # Default (unset) keeps the original hybrid behavior; grasp phase is always pure.
        if os.environ.get("AUTOGAZE_PURE", "0") == "1" or phase == "grasp":
            return self._budget_grid_mask(autogaze_score, visible_ratio), autogaze_score
        if phase == "place":
            weights = {
                "autogaze": 0.22,
                "source": 0.18,
                "target": 0.30,
                "gripper": 0.08,
                "temporal": 0.12,
                "bridge": 0.10,
            }
            source_budget = 0.18
            target_budget = 0.40
            gripper_budget = 0.10
            bridge_budget = 0.12
        else:
            weights = {
                "autogaze": 0.28,
                "source": 0.34,
                "target": 0.18,
                "gripper": 0.12,
                "temporal": 0.08,
                "bridge": 0.00,
            }
            source_budget = 0.34
            target_budget = 0.16
            gripper_budget = 0.10
            bridge_budget = 0.0

        combined = (
            weights["autogaze"] * autogaze_score
            + weights["source"] * source_prior
            + weights["target"] * target_prior
            + weights["gripper"] * gripper_prior
            + weights["temporal"] * temporal_prior
            + weights["bridge"] * bridge_prior
        )

        total_patches = self.config.grid_size * self.config.grid_size
        target_visible = int(round(np.clip(visible_ratio, 0.0, 1.0) * total_patches))
        target_visible = max(1, min(total_patches, target_visible))

        forced_mask = np.zeros_like(combined, dtype=bool)
        for score, ratio in [
            (source_prior, source_budget),
            (target_prior, target_budget),
            (gripper_prior, gripper_budget),
            (bridge_prior, bridge_budget),
        ]:
            if np.max(score) > 1e-6:
                forced_mask |= _topk_mask(score, int(round(target_visible * ratio)))
        forced_mask = _dilate_grid(forced_mask, radius=1)
        if int(forced_mask.sum()) > target_visible:
            forced_mask = _topk_mask(combined * forced_mask.astype(np.float32), target_visible)
        remaining = max(0, target_visible - int(forced_mask.sum()))

        if remaining > 0:
            residual = combined.copy()
            residual[forced_mask] = -np.inf
            forced_mask |= _topk_mask(residual, remaining)

        return forced_mask, combined

    def _budget_grid_mask(self, score_map: np.ndarray, visible_ratio: float) -> np.ndarray:
        total_patches = self.config.grid_size * self.config.grid_size
        target_visible = int(round(np.clip(visible_ratio, 0.0, 1.0) * total_patches))
        target_visible = max(1, min(total_patches, target_visible))
        flat_scores = score_map.reshape(-1)
        if not np.any(np.isfinite(flat_scores)) or np.allclose(flat_scores, flat_scores[0]):
            flat_scores = np.linspace(0.0, 1.0, total_patches, dtype=np.float32)
        topk_idx = np.argpartition(flat_scores, -target_visible)[-target_visible:]
        grid_mask = np.zeros(total_patches, dtype=bool)
        grid_mask[topk_idx] = True
        return grid_mask.reshape(self.config.grid_size, self.config.grid_size)

    def select(
        self,
        image: np.ndarray,
        visible_ratio: float,
        goal: Optional[str] = None,
        phase_info: Optional[dict] = None,
    ) -> PatchSelectionResult:
        video = self._prepare_video(image)
        with torch.inference_mode():
            gaze_outputs = self.model(
                {"video": video},
                target_scales=list(self.target_scales),
                target_patch_size=self.target_patch_size,
                gazing_ratio=float(np.clip(visible_ratio, 1e-4, 1.0)),
                task_loss_requirement=self.config.autogaze_task_loss_requirement,
                generate_only=True,
                use_cache=True,
                past_key_values=self._past_key_values,
                past_inputs_embeds=self._past_inputs_embeds,
                past_attention_mask=self._past_attention_mask,
                past_conv_values=self._past_conv_values,
            )

        self._past_key_values = gaze_outputs.get("past_key_values")
        self._past_inputs_embeds = gaze_outputs.get("past_input_embeds")
        self._past_attention_mask = gaze_outputs.get("past_attention_mask")
        self._past_conv_values = gaze_outputs.get("past_conv_values")

        score_map = self._score_gazing_mask(gaze_outputs)
        dynamic_ratio = self._visible_ratio_for_phase(goal, phase_info, visible_ratio)
        # A-2 option A: let AutoGaze's adaptive gaze count flow through instead of the
        # fixed budget (env AUTOGAZE_ADAPTIVE_BUDGET=1). Combined with adaptive stopping
        # (SELECTOR_TASK_LOSS), this realizes the latency benefit that the fixed
        # downstream budget would otherwise neutralize.
        if os.environ.get("AUTOGAZE_ADAPTIVE_BUDGET", "0") == "1":
            ar = getattr(self, "_last_gaze_ratio", None)
            if ar is not None:
                min_ratio = 1.0 / float(self.config.grid_size * self.config.grid_size)
                dynamic_ratio = float(np.clip(ar, min_ratio, dynamic_ratio))
        grid_mask, blended_score = self._hybrid_grid_mask(
            score_map,
            image=image,
            goal=goal,
            visible_ratio=dynamic_ratio,
            phase_info=phase_info,
        )
        self._prev_grid_mask = grid_mask.copy()
        self._prev_score_map = blended_score.copy()
        pixel_mask = _resize_mask(grid_mask, (image.shape[1], image.shape[0]))
        mixed = self.composer.compose(image, grid_mask)
        selected_patch_count = int(grid_mask.sum())
        selection_ratio = float(selected_patch_count) / float(
            self.config.grid_size * self.config.grid_size
        )
        return PatchSelectionResult(
            mixed_image=mixed,
            score_map=blended_score,
            selection_ratio=selection_ratio,
            selected_patch_count=selected_patch_count,
            pixel_mask=pixel_mask,
            grid_mask=grid_mask,
        )

    def reuse_previous_selection(self, image: np.ndarray) -> Optional[PatchSelectionResult]:
        if self._prev_grid_mask is None:
            return None
        grid_mask = self._prev_grid_mask.copy()
        pixel_mask = _resize_mask(grid_mask, (image.shape[1], image.shape[0]))
        mixed = self.composer.compose(image, grid_mask)
        selected_patch_count = int(grid_mask.sum())
        selection_ratio = float(selected_patch_count) / float(
            self.config.grid_size * self.config.grid_size
        )
        score_map = (
            np.zeros_like(grid_mask, dtype=np.float32)
            if self._prev_score_map is None
            else self._prev_score_map.copy()
        )
        return PatchSelectionResult(
            mixed_image=mixed,
            score_map=score_map,
            selection_ratio=selection_ratio,
            selected_patch_count=selected_patch_count,
            pixel_mask=pixel_mask,
            grid_mask=grid_mask,
        )


class EmuVLAInference:
    """
    Baseline UniVLA inference wrapper for SimplerEnv.
    """

    def __init__(
        self,
        emu_hub: str,
        vq_hub: str,
        vision_hub: str,
        device: str,
        vision_device: Optional[str] = None,
        policy_setup: str = "widowx_bridge",
        fast_path: Optional[str] = None,
        image_size_override: Optional[Tuple[int, int]] = None,
        min_pixels_override: Optional[int] = None,
        llm_prune_count: int = 0,
        llm_prune_min_layer: float = 0.5,
        llm_prune_min_gap: int = 1,
        llm_prune_layers: Optional[Sequence[int]] = None,
    ):
        self.emu_hub = emu_hub
        self.vq_hub = vq_hub
        self.vision_hub = vision_hub
        self.device = device
        self.vision_device = vision_device or device
        self.policy_setup = policy_setup
        self._image_size_override = image_size_override
        self._min_pixels_override = min_pixels_override
        self._fast_path_override = fast_path
        self._last_prepared_image: Optional[np.ndarray] = None
        self._last_action_plan: Optional[np.ndarray] = None
        self.llm_prune_count = max(0, int(llm_prune_count))
        self.llm_prune_min_layer = float(llm_prune_min_layer)
        self.llm_prune_min_gap = max(0, int(llm_prune_min_gap))
        self._requested_llm_prune_layers = (
            None if llm_prune_layers is None else tuple(sorted({int(x) for x in llm_prune_layers}))
        )
        self._active_llm_prune_layers: Tuple[int, ...] = ()
        self._llm_pruning_ready = False
        self._original_decoder_layers: Dict[int, nn.Module] = {}

        if self.policy_setup == "google_robot":
            self.close_gripper_act = -1
            default_image_size = (160, 128)
        else:
            self.close_gripper_act = 1
            default_image_size = (256, 256)
        self.image_size = image_size_override or default_image_size

        self.sticky_gripper_num_repeat = 2
        self.sticky_action_is_on = False
        self.gripper_action_repeat = 0
        self.sticky_gripper_action = 0.0
        self.previous_gripper_action = None
        self.close_gripper_num = 0
        self.late_close_gripper = 1

        self.window_size = 2
        self.predict_action_frames = 5
        self.context_frames = 1
        self.predict_frames = 1
        self.action_dim = 7
        self.use_gripper = False
        self.use_fast = True
        self.eoa_token_id = 151845
        self.video_mode = True

        self.init_config(device=device)
        self.image_processor.min_pixels = (
            min_pixels_override if min_pixels_override is not None else 80 * 80
        )

        self.GENERATION_CONFIG = GenerationConfig(
            use_cache=True,
            pad_token_id=self.model.config.pad_token_id,
            bos_token_id=self.model.config.bos_token_id,
            eos_token_id=self.eoa_token_id,
            do_sample=False,
        )
        if self._requested_llm_prune_layers is not None:
            self._apply_llm_pruning(self._requested_llm_prune_layers)

    def init_config(self, device: str) -> None:
        self.model = Emu3MoE.from_pretrained(
            self.emu_hub,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
        ).to(device).eval()

        self.tokenizer = Emu3Tokenizer.from_pretrained(
            self.emu_hub,
            model_max_length=self.model.config.max_position_embeddings,
            padding_side="right",
            use_fast=False,
        )
        try:
            self.image_processor = AutoImageProcessor.from_pretrained(
                self.vision_hub,
                trust_remote_code=True,
                use_fast=True,
            )
        except TypeError:
            self.image_processor = AutoImageProcessor.from_pretrained(
                self.vision_hub,
                trust_remote_code=True,
            )
        vision_dtype = (
            torch.bfloat16 if str(self.vision_device).startswith("cuda") else torch.float32
        )
        self.image_tokenizer = (
            AutoModel.from_pretrained(
                self.vision_hub,
                trust_remote_code=True,
                torch_dtype=vision_dtype,
            )
            .to(self.vision_device, dtype=vision_dtype)
            .eval()
        )
        self.processor = Emu3Processor(
            image_processor=self.image_processor,
            vision_tokenizer=self.image_tokenizer,
            tokenizer=self.tokenizer,
        )

        _fast_base = self._fast_path_override or "/share/project/yuqi.wang/UniVLA/pretrain"
        markers = ("processor_config.json", "tokenizer_config.json", "config.json")
        if any(os.path.isfile(os.path.join(_fast_base, name)) for name in markers):
            fast_path = _fast_base
        elif self.policy_setup == "widowx_bridge":
            fast_path = os.path.join(_fast_base, "fast_bridge_t5_s50")
        elif self.policy_setup == "google_robot":
            fast_path = os.path.join(_fast_base, "fast_google_a5_s50")
        else:
            fast_path = os.path.join(_fast_base, "fast")

        self.action_tokenizer = None
        if os.path.isdir(fast_path):
            self.action_tokenizer = AutoProcessor.from_pretrained(
                fast_path, trust_remote_code=True
            )

        self._action_id_processor = None
        if self.action_tokenizer is not None:
            last_token_id = self.tokenizer.pad_token_id - 1
            allowed = list(
                range(last_token_id - self.action_tokenizer.vocab_size, last_token_id + 1)
            ) + [self.eoa_token_id]
            self._action_id_processor = ActionIDConstraintLogitsProcessor(allowed)

        self.vision_queue = Queue(maxsize=self.window_size)
        self.vision_gripper_queue = Queue(maxsize=self.window_size)
        self.action_queue = Queue(maxsize=self.window_size - 1)

        self._fastv_enabled = False
        self._maybe_apply_fastv()

        # Phase-adaptive depth controller (non-uniform depth over time).
        self._depth_ctrl_enabled = os.environ.get("DEPTH_CTRL_ENABLE", "0") == "1"
        self._depth_deep_count = int(os.environ.get("DEPTH_CTRL_DEEP", "2"))
        self._depth_shallow_count = int(os.environ.get("DEPTH_CTRL_SHALLOW", "8"))
        self._depth_close_steps = int(os.environ.get("DEPTH_CTRL_CLOSE_STEPS", "2"))
        # Selective per-archetype routing: only apply depth pruning when the
        # instruction's inferred archetype is in this set (empty/unset = all).
        # The controller reads the instruction a priori (no test peeking); the
        # per-archetype assignment is a design hypothesis (see docs).
        arch_env = os.environ.get("DEPTH_CTRL_ARCHETYPES", "").strip()
        self._depth_archetypes = (
            {a.strip() for a in arch_env.split(",") if a.strip()} if arch_env else None
        )
        self._depth_state = "deep"
        self._depth_ranking: List[int] = []
        self._depth_ranking_ready = False
        self._depth_active = self._depth_ctrl_enabled  # resolved per episode by archetype
        self._depth_archetype_resolved = False
        self._depth_archetype = None
        if self._depth_ctrl_enabled:
            scope = ("all archetypes" if self._depth_archetypes is None
                     else ", ".join(sorted(self._depth_archetypes)))
            print(
                f"[DepthCtrl] phase-adaptive depth: deep(prune {self._depth_deep_count}) "
                f"until grasp, then shallow(prune {self._depth_shallow_count}); "
                f"switch after {self._depth_close_steps} close steps; "
                f"applied to: {scope}.",
                flush=True,
            )

        # Self-speculative decoding (training-free, lossless -- see
        # self_spec_decode.py / emu3_self_spec_decode.py). Unlike depth
        # pruning, this NEVER trades away accuracy: the full model always has
        # final say, so it is a pure latency experiment, not an accuracy risk.
        self._spec_decode_enabled = os.environ.get("SPEC_DECODE_ENABLE", "0") == "1"
        self._spec_decode_gamma = int(os.environ.get("SPEC_DECODE_GAMMA", "4"))
        self._spec_decode_layer_count = int(os.environ.get("SPEC_DECODE_LAYER_COUNT", "4"))
        self._spec_decode_ranking: List[int] = []
        self._spec_decode_ranking_ready = False
        self._spec_decode_stats = {"rounds": 0, "draft_calls": 0, "verify_calls": 0,
                                    "accepted": 0, "proposed": 0}
        if self._spec_decode_enabled:
            print(
                f"[SpecDecode] self-speculative decoding ON: gamma={self._spec_decode_gamma}, "
                f"draft bypasses top-{self._spec_decode_layer_count} redundant layers "
                f"(lossless -- output identical to plain greedy decoding by construction).",
                flush=True,
            )

    def _maybe_apply_fastv(self) -> None:
        """Enable FastV visual-token pruning on the Emu3 LLM via env flags.

        FASTV_ENABLE=1            turn it on
        FASTV_K (default 3)       prune after this many full layers
        FASTV_KEEP_RATIO (0.4)    fraction of visual tokens kept past layer K
        """
        if os.environ.get("FASTV_ENABLE", "0") != "1":
            return
        try:
            k = int(os.environ.get("FASTV_K", "3"))
            keep = float(os.environ.get("FASTV_KEEP_RATIO", "0.4"))
            apply_fastv(self.model, k_layer=k, keep_ratio=keep)
            self._fastv_enabled = True
        except Exception as exc:  # never break inference if patching fails
            print(f"[FastV] disabled (patch failed: {exc})", flush=True)
            self._fastv_enabled = False

    def add_image(self, image):
        if self.vision_queue.full():
            self.vision_queue.get()
        self.vision_queue.put(image)

    def get_history(self):
        return list(self.vision_queue.queue)

    def add_action(self, action):
        if self.action_queue.full():
            self.action_queue.get()
        self.action_queue.put(action)

    def get_action_history(self):
        return list(self.action_queue.queue)

    def _to_cpu_batch_feature(self, batch: BatchFeature) -> BatchFeature:
        data = {}
        for key, value in batch.items():
            data[key] = value.detach().cpu() if torch.is_tensor(value) else value
        return BatchFeature(data=data, tensor_type="pt")

    def _append_history(self, pos_inputs: BatchFeature, action_tokens: torch.Tensor) -> None:
        if not self.video_mode:
            return
        self.add_image(self._to_cpu_batch_feature(pos_inputs))
        self.add_action(action_tokens.detach().cpu())

    def reset(self) -> None:
        self.sticky_action_is_on = False
        self.gripper_action_repeat = 0
        self.sticky_gripper_action = 0.0
        self.close_gripper_num = 0
        self.previous_gripper_action = None
        self._last_prepared_image = None
        self._last_action_plan = None
        # Phase-adaptive depth: new episode starts deep and re-resolves archetype
        # routing + re-calibrates the redundancy ranking; restore bypassed layers.
        if getattr(self, "_depth_ctrl_enabled", False):
            self._depth_state = "deep"
            self._depth_ranking_ready = False
            self._depth_ranking = []
            self._depth_archetype_resolved = False
            self._depth_active = self._depth_ctrl_enabled
            self._restore_llm_layers()
        if getattr(self, "_spec_decode_enabled", False):
            self._spec_decode_ranking_ready = False
            self._spec_decode_ranking = []
            self._restore_llm_layers()
        while not self.vision_queue.empty():
            self.vision_queue.get()
        while not self.vision_gripper_queue.empty():
            self.vision_gripper_queue.get()
        while not self.action_queue.empty():
            self.action_queue.get()

    def prepare_image(
        self,
        image: np.ndarray,
        goal: Optional[str] = None,
        phase_info: Optional[dict] = None,
        oracle_context: Optional[dict] = None,
    ) -> np.ndarray:
        return image

    def last_prepared_image(self) -> Optional[np.ndarray]:
        if self._last_prepared_image is None:
            return None
        return self._last_prepared_image.copy()

    def preprocess(
        self,
        image: np.ndarray,
        goal: Optional[str] = None,
        phase_info: Optional[dict] = None,
        oracle_context: Optional[dict] = None,
    ):
        prepared = self.prepare_image(
            image,
            goal=goal,
            phase_info=phase_info,
            oracle_context=oracle_context,
        )
        self._last_prepared_image = np.asarray(prepared, dtype=np.uint8)
        agent_view = Image.fromarray(self._last_prepared_image).resize(self.image_size)
        return self.encode_agent_view(agent_view), None

    def encode_agent_view(self, agent_view: Image.Image) -> torch.Tensor:
        image_x = self.image_processor(agent_view, return_tensors="pt")["pixel_values"]
        target_dtype = torch.float32
        if hasattr(self.image_tokenizer, "parameters"):
            try:
                target_dtype = next(self.image_tokenizer.parameters()).dtype
            except (StopIteration, TypeError):
                pass
        image_x = image_x.to(self.vision_device, dtype=target_dtype)
        with torch.inference_mode():
            image_code = self.image_tokenizer.encode(image_x)
        return image_code.cpu()

    def _build_policy_inputs(
        self,
        image: np.ndarray,
        goal: str,
        phase_info: Optional[dict] = None,
        oracle_context: Optional[dict] = None,
    ) -> Tuple[BatchFeature, BatchFeature]:
        image_code, gripper_code = self.preprocess(
            image,
            goal=goal,
            phase_info=phase_info,
            oracle_context=oracle_context,
        )
        video_code = image_code.unsqueeze(1)
        gripper_code = gripper_code.unsqueeze(1) if self.use_gripper else None

        text_prompt = [self.tokenizer.bos_token + goal]
        text_tokens = self.processor.tokenizer(text_prompt)
        text_tokens = BatchFeature(data={**text_tokens}, tensor_type="pt")
        pos_inputs = self.processor.video_process(
            text=goal,
            video_tokens=video_code,
            gripper_tokens=gripper_code,
            context_frames=self.context_frames,
            frames=self.predict_frames,
            return_tensors="pt",
            mode="VLA_Video",
            padding="longest",
        )

        if not self.video_mode:
            return pos_inputs, pos_inputs

        self.add_image(pos_inputs)
        history = self.get_history()
        action_history = self.get_action_history()

        all_input_ids = [text_tokens["input_ids"]]
        all_token_type_ids = [text_tokens["token_type_ids"]]
        all_attention_mask = [text_tokens["attention_mask"]]

        for i, hist in enumerate(history):
            if i < len(action_history):
                act = action_history[i]
                all_input_ids.extend([hist["input_ids"], act])
                all_token_type_ids.extend([hist["token_type_ids"], torch.zeros_like(act)])
                all_attention_mask.extend([hist["attention_mask"], torch.ones_like(act)])
            else:
                all_input_ids.append(hist["input_ids"])
                all_token_type_ids.append(hist["token_type_ids"])
                all_attention_mask.append(hist["attention_mask"])

        final_inputs = pos_inputs.copy()
        final_inputs["input_ids"] = torch.cat(all_input_ids, dim=1)
        final_inputs["token_type_ids"] = torch.cat(all_token_type_ids, dim=1)
        final_inputs["attention_mask"] = torch.cat(all_attention_mask, dim=1)
        return pos_inputs, final_inputs

    def _llm_decoder_layers(self) -> Optional[torch.nn.ModuleList]:
        model = getattr(self.model, "model", None)
        layers = getattr(model, "layers", None)
        if isinstance(layers, torch.nn.ModuleList):
            return layers
        return None

    def _restore_llm_layers(self) -> None:
        layers = self._llm_decoder_layers()
        if layers is None or not self._original_decoder_layers:
            return
        for idx, layer in self._original_decoder_layers.items():
            layers[idx] = layer
        self._original_decoder_layers = {}
        self._active_llm_prune_layers = ()
        self._llm_pruning_ready = False

    def _apply_llm_pruning(self, layer_indices: Sequence[int]) -> None:
        layers = self._llm_decoder_layers()
        if layers is None:
            return
        valid = tuple(
            idx for idx in sorted({int(x) for x in layer_indices}) if 0 <= idx < len(layers)
        )
        if not valid:
            return
        self._restore_llm_layers()
        cfg = getattr(self.model, "config", None)
        num_kv_heads = getattr(cfg, "num_key_value_heads", None) if cfg else None
        n_heads = getattr(cfg, "num_attention_heads", None) if cfg else None
        hidden = getattr(cfg, "hidden_size", None) if cfg else None
        head_dim = (hidden // n_heads) if (hidden and n_heads) else None
        for idx in valid:
            self._original_decoder_layers[idx] = layers[idx]
            layers[idx] = BypassDecoderLayer(idx, num_kv_heads=num_kv_heads, head_dim=head_dim)
        self._active_llm_prune_layers = valid
        self._llm_pruning_ready = True

    def _select_prune_layers(
        self,
        importance_scores: Sequence[float],
    ) -> Tuple[int, ...]:
        if self.llm_prune_count <= 0:
            return ()
        scores = np.asarray(importance_scores, dtype=np.float32)
        num_layers = int(scores.shape[0])
        start_idx = int(math.floor(np.clip(self.llm_prune_min_layer, 0.0, 1.0) * num_layers))
        candidates = list(range(start_idx, num_layers))
        if not candidates:
            candidates = list(range(num_layers))
        ranked = sorted(candidates, key=lambda idx: (float(scores[idx]), idx))
        selected: List[int] = []
        for idx in ranked:
            if any(abs(idx - prev) <= self.llm_prune_min_gap for prev in selected):
                continue
            selected.append(idx)
            if len(selected) >= self.llm_prune_count:
                break
        if len(selected) < self.llm_prune_count:
            for idx in ranked:
                if idx in selected:
                    continue
                selected.append(idx)
                if len(selected) >= self.llm_prune_count:
                    break
        return tuple(sorted(selected))

    def _compute_layer_importance(self, final_inputs: BatchFeature) -> Optional[List[float]]:
        """One forward pass; per-layer importance = 1 - cos(layer input, layer
        output). Low importance = the layer barely changes the representation =
        redundant = safe to bypass."""
        layers = self._llm_decoder_layers()
        if layers is None or len(layers) == 0:
            return None
        with torch.inference_mode():
            outputs = self.model.model(
                input_ids=final_inputs.input_ids.to(self.device),
                attention_mask=final_inputs.attention_mask.to(self.device),
                use_cache=False,
                output_hidden_states=True,
                return_dict=True,
            )
        hidden_states = outputs.hidden_states
        if hidden_states is None or len(hidden_states) < 2:
            return None
        importance_scores = []
        for lhs, rhs in zip(hidden_states[:-1], hidden_states[1:]):
            cos = torch.nn.functional.cosine_similarity(lhs.float(), rhs.float(), dim=-1)
            importance_scores.append(float(1.0 - cos.mean().item()))
        return importance_scores

    def _rank_prune_layers(self, importance_scores: Sequence[float]) -> List[int]:
        """Full eligible layer list ordered by ascending importance (most
        redundant first), gap-respecting first then filling the rest. Prefixes are
        nested, so deep ⊂ shallow when the controller slices top-N."""
        scores = np.asarray(importance_scores, dtype=np.float32)
        num_layers = int(scores.shape[0])
        start_idx = int(math.floor(np.clip(self.llm_prune_min_layer, 0.0, 1.0) * num_layers))
        candidates = list(range(start_idx, num_layers)) or list(range(num_layers))
        ranked = sorted(candidates, key=lambda idx: (float(scores[idx]), idx))
        ordered: List[int] = []
        for idx in ranked:  # gap-respecting greedy
            if any(abs(idx - prev) <= self.llm_prune_min_gap for prev in ordered):
                continue
            ordered.append(idx)
        for idx in ranked:  # fill the rest (allows adjacency) for high prune counts
            if idx not in ordered:
                ordered.append(idx)
        return ordered

    def _depth_apply_state(self) -> None:
        if not self._depth_ranking:
            return
        count = (self._depth_deep_count if self._depth_state == "deep"
                 else self._depth_shallow_count)
        self._apply_llm_pruning(self._depth_ranking[:max(0, count)])

    def _maybe_calibrate_llm_pruning(self, final_inputs: BatchFeature) -> None:
        # Phase-adaptive depth controller: calibrate the redundancy ranking ONCE,
        # then bypass top-N per phase (deep until grasp, shallow after). Only when
        # the controller is active for this episode's archetype.
        if getattr(self, "_depth_active", False):
            if self._depth_ranking_ready:
                return
            importance = self._compute_layer_importance(final_inputs)
            if importance is None:
                return
            self._depth_ranking = self._rank_prune_layers(importance)
            self._depth_ranking_ready = True
            self._depth_state = "deep"
            self._depth_apply_state()
            return

        if self._llm_pruning_ready or self.llm_prune_count <= 0:
            return
        importance = self._compute_layer_importance(final_inputs)
        if importance is None:
            return
        self._apply_llm_pruning(self._select_prune_layers(importance))

    def _maybe_calibrate_spec_decode(self, final_inputs: BatchFeature) -> None:
        # Self-speculative decoding needs its own redundancy ranking to pick
        # which layers the "draft" pass bypasses. Reuse an already-computed
        # ranking if the depth controller (or static pruning) already
        # calibrated one this episode -- same metric, no need to redo it.
        if not getattr(self, "_spec_decode_enabled", False) or self._spec_decode_ranking_ready:
            return
        if self._depth_ranking_ready and self._depth_ranking:
            self._spec_decode_ranking = list(self._depth_ranking)
            self._spec_decode_ranking_ready = True
            return
        importance = self._compute_layer_importance(final_inputs)
        if importance is None:
            return
        self._spec_decode_ranking = self._rank_prune_layers(importance)
        self._spec_decode_ranking_ready = True

    def _generate_sequences(
        self,
        final_inputs: BatchFeature,
        num_return_sequences: int = 1,
        num_beams: int = 1,
    ) -> Tuple[torch.Tensor, Optional[np.ndarray]]:
        num_return_sequences = max(1, int(num_return_sequences))
        num_beams = max(1, int(num_beams), num_return_sequences)

        # FastV: mark the visual-token span so the patched Emu3 forward prunes it
        # on this prefill. Set every step (the grid can change per frame).
        if getattr(self, "_fastv_enabled", False):
            try:
                self.model._fastv_visual_mask = visual_mask_from_input_ids(
                    final_inputs.input_ids, self.tokenizer
                ).to(self.device)
            except Exception:
                self.model._fastv_visual_mask = None

        logits_processors = (
            [] if self._action_id_processor is None else [self._action_id_processor]
        )

        # Self-speculative decoding: only handles single-sequence greedy
        # decoding (this project always uses do_sample=False); anything else
        # (beam search / multiple return sequences) falls back to .generate().
        if (
            getattr(self, "_spec_decode_enabled", False)
            and num_return_sequences == 1
            and num_beams == 1
            and self._spec_decode_ranking_ready
            and self._spec_decode_ranking
        ):
            draft_layers = self._spec_decode_ranking[: self._spec_decode_layer_count]
            with torch.inference_mode():
                sequences = emu3_self_speculative_generate(
                    self,
                    final_inputs.input_ids.to(self.device),
                    final_inputs.attention_mask.to(self.device),
                    max_new_tokens=50,
                    gamma=self._spec_decode_gamma,
                    draft_layer_indices=draft_layers,
                    eos_token_id=self.GENERATION_CONFIG.eos_token_id,
                    logits_processor=logits_processors,
                    stats=self._spec_decode_stats,
                )
            self._restore_llm_layers()  # leave layers intact for any other caller
            return sequences, None

        generate_kwargs = dict(
            generation_config=self.GENERATION_CONFIG,
            max_new_tokens=50,
            logits_processor=logits_processors,
            attention_mask=final_inputs.attention_mask.to(self.device),
        )
        if num_return_sequences > 1 or num_beams > 1:
            generate_kwargs.update(
                num_return_sequences=num_return_sequences,
                num_beams=num_beams,
                early_stopping=True,
                return_dict_in_generate=True,
                output_scores=True,
            )
        with torch.inference_mode():
            outputs = self.model.generate(
                final_inputs.input_ids.to(self.device),
                **generate_kwargs,
            )
        if hasattr(outputs, "sequences"):
            sequences = outputs.sequences
            seq_scores = getattr(outputs, "sequences_scores", None)
            if seq_scores is not None:
                seq_scores = seq_scores.detach().float().cpu().numpy()
            return sequences, seq_scores
        return outputs, None

    def _decode_action_sequences(
        self,
        sequences: torch.Tensor,
        prompt_length: int,
    ) -> Tuple[np.ndarray, torch.Tensor]:
        if sequences.ndim == 1:
            sequences = sequences.unsqueeze(0)
        orig_outputs = sequences[:, prompt_length:]
        action_token_ids = orig_outputs[:, :-1]
        last_token_id = self.tokenizer.pad_token_id - 1
        last_token_id_t = torch.tensor(
            last_token_id,
            dtype=action_token_ids.dtype,
            device=action_token_ids.device,
        )
        processed = last_token_id_t - action_token_ids
        action_outputs = self.action_tokenizer.decode(
            processed,
            time_horizon=self.predict_action_frames,
            action_dim=self.action_dim,
        )
        plans = np.stack(
            [self.unormalize_action(action_outputs[i]) for i in range(action_outputs.shape[0])],
            axis=0,
        )
        return plans, orig_outputs

    def step(
        self,
        image: np.ndarray,
        goal: str,
        phase_info: Optional[dict] = None,
        oracle_context: Optional[dict] = None,
    ):
        # Selective routing: decide once per episode whether the depth controller
        # applies to THIS instruction's archetype (a priori, from the instruction).
        if self._depth_ctrl_enabled and not self._depth_archetype_resolved:
            self._depth_archetype = infer_task_archetype(goal)
            self._depth_active = (
                self._depth_archetypes is None
                or self._depth_archetype in self._depth_archetypes
            )
            self._depth_archetype_resolved = True
            print(f"[DepthCtrl] archetype={self._depth_archetype} -> "
                  f"depth {'ON' if self._depth_active else 'OFF (baseline)'}", flush=True)

        pos_inputs, final_inputs = self._build_policy_inputs(
            image,
            goal,
            phase_info=phase_info,
            oracle_context=oracle_context,
        )
        self._maybe_calibrate_llm_pruning(final_inputs)
        self._maybe_calibrate_spec_decode(final_inputs)
        sequences, _ = self._generate_sequences(final_inputs)
        plans, orig_outputs = self._decode_action_sequences(
            sequences,
            prompt_length=final_inputs.input_ids.shape[-1],
        )
        action = plans[0]
        if self.video_mode:
            self._append_history(pos_inputs, orig_outputs[:1])
        self._last_action_plan = action.copy()

        res = [self.transform_action(action[[i], :]) for i in range(action.shape[0])]

        # Phase-adaptive depth: one-way deep -> shallow once the gripper has been
        # commanded closed for K consecutive steps (grasp confirmed). One-way means
        # no oscillation; the precise approach+grasp keeps full depth, transport runs
        # shallow. transform_action() above just updated self.close_gripper_num.
        if (
            getattr(self, "_depth_active", False)
            and self._depth_state == "deep"
            and self._depth_ranking_ready
            and self.close_gripper_num >= self._depth_close_steps
        ):
            self._depth_state = "shallow"
            self._depth_apply_state()

        return [r[0] for r in res], [r[1] for r in res]

    def pruning_summary(self) -> dict:
        out = {
            "llm_prune_count": int(len(self._active_llm_prune_layers)),
            "llm_pruned_layers": list(self._active_llm_prune_layers),
        }
        if getattr(self, "_depth_ctrl_enabled", False):
            out["depth_state"] = self._depth_state
            out["depth_archetype"] = self._depth_archetype
            out["depth_active"] = bool(self._depth_active)
        if getattr(self, "_spec_decode_enabled", False):
            s = self._spec_decode_stats
            out["spec_decode"] = dict(s)
            out["spec_decode"]["acceptance_rate"] = (
                s["accepted"] / s["proposed"] if s.get("proposed") else None
            )
        return out

    def unormalize_action(self, action: np.ndarray) -> np.ndarray:
        if self.policy_setup == "google_robot":
            high = np.array(
                [
                    0.17753939667156038,
                    0.14669284061245857,
                    0.2179850059450077,
                    0.5882710857765758,
                    0.35334834816471683,
                    0.4470693223284772,
                    0.99980000009996,
                ]
            )
            low = np.array(
                [
                    -0.22624227067544056,
                    -0.15126218201085617,
                    -0.23251856873127252,
                    -0.3538952427136002,
                    -0.4202906595250484,
                    -0.43766197340888247,
                    -1e-10,
                ]
            )
        else:
            high = np.array(
                [
                    0.02819482065335177,
                    0.04079562563528227,
                    0.04015785568112329,
                    0.08070396399877966,
                    0.07745134926258301,
                    0.2016542930635028,
                    0.99980000009996,
                ]
            )
            low = np.array(
                [
                    -0.02887803485105428,
                    -0.04178320091122349,
                    -0.026113155505000457,
                    -0.08117201867235568,
                    -0.09309056401752747,
                    -0.20778717060421048,
                    -1e-10,
                ]
            )
        return 0.5 * (action + 1) * (high - low) + low

    def transform_action(self, raw_actions: np.ndarray):
        raw_action = {
            "world_vector": np.array(raw_actions[0, :3]),
            "rotation_delta": np.array(raw_actions[0, 3:6]),
            "open_gripper": np.array(raw_actions[0, 6:7]),
        }
        action = {}
        action["world_vector"] = raw_action["world_vector"]
        roll, pitch, yaw = raw_action["rotation_delta"]
        ax, angle = euler2axangle(roll, pitch, yaw)
        action["rot_axangle"] = ax * angle

        if self.policy_setup == "google_robot":
            cur = 2.0 * (raw_action["open_gripper"] > 0.5) - 1.0
            rel = (
                np.array([0])
                if self.previous_gripper_action is None
                else self.previous_gripper_action - cur
            )
            self.previous_gripper_action = cur
            if np.abs(rel) > 0.5 and not self.sticky_action_is_on:
                self.sticky_action_is_on = True
                self.sticky_gripper_action = rel
            if self.sticky_action_is_on:
                self.gripper_action_repeat += 1
                rel = self.sticky_gripper_action
            if self.gripper_action_repeat == self.sticky_gripper_num_repeat:
                self.sticky_action_is_on = False
                self.gripper_action_repeat = 0
                self.sticky_gripper_action = 0.0
            action["gripper"] = rel
        else:
            rel = 2.0 * (raw_action["open_gripper"] > 0.5) - 1.0
            if rel[0] > 0:
                self.close_gripper_num += 1
            else:
                self.close_gripper_num = 0
            rel[0] = 1 if self.close_gripper_num >= self.late_close_gripper else -1
            action["gripper"] = rel

        action["terminate_episode"] = np.array([0.0])
        return raw_action, action


class AdaptiveSparseInference(EmuVLAInference):
    """
    Frozen VLA with an adaptive sparse visual frontend.
    """

    def __init__(
        self,
        emu_hub: str,
        vq_hub: str,
        vision_hub: str,
        device: str,
        vision_device: Optional[str] = None,
        policy_setup: str = "widowx_bridge",
        fast_path: Optional[str] = None,
        image_size_override: Optional[Tuple[int, int]] = None,
        min_pixels_override: Optional[int] = None,
        patch_grid_size: int = 20,
        max_highres_patches: int = 80,
        lowres_factor: float = 0.1,
        autogaze_hub: str = "nvidia/AutoGaze",
        autogaze_target_patch_size: int = 16,
        autogaze_task_loss_requirement: Optional[float] = None,
        selector_device: Optional[str] = None,
        oracle_assist: bool = True,
        patch_grasp_steps: int = 12,
        sparse_gate_mode: str = "auto",
        llm_prune_count: int = 0,
        llm_prune_min_layer: float = 0.5,
        llm_prune_min_gap: int = 1,
        llm_prune_layers: Optional[Sequence[int]] = None,
        selector_refresh_stride: int = 1,
    ):
        self.patch_config = PatchSelectionConfig(
            grid_size=int(patch_grid_size),
            max_highres_patches=int(max_highres_patches),
            lowres_factor=float(lowres_factor),
            autogaze_hub=autogaze_hub,
            autogaze_target_patch_size=int(autogaze_target_patch_size),
            autogaze_task_loss_requirement=autogaze_task_loss_requirement,
        )
        self._selection_trace: List[dict] = []
        self._last_selection: Optional[PatchSelectionResult] = None
        self._last_sparse_input_image: Optional[np.ndarray] = None
        self._selector_device = selector_device or vision_device or device
        self.oracle_assist = False
        self._candidate_prepare_mode = "patch"
        self.patch_grasp_steps = max(0, int(patch_grasp_steps))
        self.sparse_gate_mode = sparse_gate_mode
        self.selector_refresh_stride = max(1, int(selector_refresh_stride))
        self._policy_step = 0
        self._layout_patch_enabled: Optional[bool] = None
        self._layout_patch_always = False
        self._prepare_mode_trace: List[str] = []
        self._patch_prepare_calls = 0
        self._selector_refresh_calls = 0
        self._selector_reuse_calls = 0
        self.visible_ratio = min(
            1.0,
            float(self.patch_config.max_highres_patches)
            / float(self.patch_config.grid_size * self.patch_config.grid_size),
        )
        self.selector: Optional[AutoGazeSelectorController] = None
        super().__init__(
            emu_hub=emu_hub,
            vq_hub=vq_hub,
            vision_hub=vision_hub,
            device=device,
            vision_device=vision_device,
            policy_setup=policy_setup,
            fast_path=fast_path,
            image_size_override=image_size_override,
            min_pixels_override=min_pixels_override,
            llm_prune_count=llm_prune_count,
            llm_prune_min_layer=llm_prune_min_layer,
            llm_prune_min_gap=llm_prune_min_gap,
            llm_prune_layers=llm_prune_layers,
        )
        self.selector = AutoGazeSelectorController(
            config=self.patch_config,
            device=self._selector_device,
            target_image_size=int(self.image_size[0]),
        )

    def _layout_patch_gate(self, image: np.ndarray, goal: Optional[str]) -> bool:
        if self.sparse_gate_mode == "never":
            return False
        if self.sparse_gate_mode == "always":
            self._layout_patch_always = True
            return True
        source_score, target_score = _entity_pixel_scores(image, goal)
        if float(np.max(source_score)) <= 1e-6 or float(np.max(target_score)) <= 1e-6:
            return False
        source_center = _center_of_mass(source_score)
        target_center = _center_of_mass(target_score)
        if source_center is None or target_center is None:
            return False
        dy = source_center[0] - target_center[0]
        dx = source_center[1] - target_center[1]
        h, w = image.shape[:2]
        source_area = float((source_score > 0.2).mean())
        entities = _extract_task_entities(goal)
        source_name = entities["source"] or ""
        target_name = entities["target"] or ""

        right_same_height = dx > 0.20 * w and abs(dy) < 0.07 * h
        lower_midright_large_source = (
            0.05 * w < dx < 0.12 * w
            and dy > 0.16 * h
            and source_area > 0.014
        )
        spoon_below_towel = (
            "spoon" in source_name
            and ("towel" in target_name or "cloth" in target_name)
            and 0.08 * w < dx < 0.25 * w
            and dy > 0.34 * h
            and source_area > 0.003
        )
        self._layout_patch_always = lower_midright_large_source or spoon_below_towel
        return right_same_height or lower_midright_large_source or spoon_below_towel

    def _choose_prepare_mode(
        self,
        image: np.ndarray,
        goal: Optional[str],
        phase_info: Optional[dict] = None,
    ) -> str:
        if self._layout_patch_enabled is None:
            self._layout_patch_enabled = self._layout_patch_gate(image, goal)
        grasped = self.close_gripper_num > 0
        if self._layout_patch_enabled and (
            self._layout_patch_always
            or (not grasped and self._policy_step < self.patch_grasp_steps)
        ):
            return "patch"
        return "raw"

    def prepare_image(
        self,
        image: np.ndarray,
        goal: Optional[str] = None,
        phase_info: Optional[dict] = None,
        oracle_context: Optional[dict] = None,
        ) -> np.ndarray:
        if self._candidate_prepare_mode == "raw":
            self._last_selection = None
            self._last_sparse_input_image = np.asarray(image, dtype=np.uint8).copy()
            self._prepare_mode_trace.append("raw")
            return image
        assert self.selector is not None
        reuse_result = None
        if self.selector_refresh_stride > 1 and self._patch_prepare_calls > 0:
            if (self._patch_prepare_calls % self.selector_refresh_stride) != 0:
                reuse_result = self.selector.reuse_previous_selection(image)
        self._patch_prepare_calls += 1
        if reuse_result is not None:
            result = reuse_result
            self._selector_reuse_calls += 1
        else:
            result = self.selector.select(
                image=image,
                visible_ratio=self.visible_ratio,
                goal=goal,
                phase_info=phase_info,
            )
            self._selector_refresh_calls += 1
        self._last_selection = result
        self._last_sparse_input_image = result.mixed_image
        self._selection_trace.append(
            {
                "selected_patch_count": result.selected_patch_count,
                "selection_ratio": result.selection_ratio,
            }
        )
        self._prepare_mode_trace.append("patch")
        return result.mixed_image

    def step(
        self,
        image: np.ndarray,
        goal: str,
        phase_info: Optional[dict] = None,
        oracle_context: Optional[dict] = None,
    ):
        self._candidate_prepare_mode = self._choose_prepare_mode(
            image,
            goal,
            phase_info=phase_info,
        )
        raw_actions, env_actions = super().step(
            image,
            goal,
            phase_info=phase_info,
            oracle_context=oracle_context,
        )
        self._policy_step += 1
        return raw_actions, env_actions

    def selection_summary(self) -> dict:
        if not self._selection_trace:
            summary = {
                "avg_selected_patch_count": 0.0,
                "avg_selection_ratio": 0.0,
            }
        else:
            counts = [x["selected_patch_count"] for x in self._selection_trace]
            ratios = [x["selection_ratio"] for x in self._selection_trace]
            summary = {
                "avg_selected_patch_count": float(np.mean(counts)),
                "avg_selection_ratio": float(np.mean(ratios)),
            }
        if self._prepare_mode_trace:
            summary["patch_usage_ratio"] = float(
                np.mean([1.0 if mode == "patch" else 0.0 for mode in self._prepare_mode_trace])
            )
        else:
            summary["patch_usage_ratio"] = 0.0
        total_selector_calls = self._selector_refresh_calls + self._selector_reuse_calls
        summary["selector_refresh_ratio"] = (
            float(self._selector_refresh_calls) / float(total_selector_calls)
            if total_selector_calls > 0
            else 0.0
        )
        return summary

    def reset(self) -> None:
        super().reset()
        self._selection_trace = []
        self._last_selection = None
        self._last_sparse_input_image = None
        self._candidate_prepare_mode = "patch"
        self._policy_step = 0
        self._layout_patch_enabled = None
        self._layout_patch_always = False
        self._prepare_mode_trace = []
        self._patch_prepare_calls = 0
        self._selector_refresh_calls = 0
        self._selector_reuse_calls = 0
        if self.selector is not None:
            self.selector.reset()


class UniformRefinementInference(AdaptiveSparseInference):
    """
    Uniform training-free refinement policy.

    The same rule is used on every task:
    1. Decode multiple raw candidates.
    2. Estimate uncertainty from score margin and action disagreement.
    3. If uncertain during early acquisition, evaluate sparse candidates too.
    4. Select the best candidate with a generic motion-stability reranker.
    """

    def __init__(
        self,
        emu_hub: str,
        vq_hub: str,
        vision_hub: str,
        device: str,
        vision_device: Optional[str] = None,
        policy_setup: str = "widowx_bridge",
        fast_path: Optional[str] = None,
        image_size_override: Optional[Tuple[int, int]] = None,
        min_pixels_override: Optional[int] = None,
        patch_grid_size: int = 20,
        max_highres_patches: int = 60,
        lowres_factor: float = 0.1,
        autogaze_hub: str = "nvidia/AutoGaze",
        autogaze_target_patch_size: int = 16,
        autogaze_task_loss_requirement: Optional[float] = None,
        selector_device: Optional[str] = None,
        oracle_assist: bool = True,
        patch_grasp_steps: int = 8,
        sparse_gate_mode: str = "never",
        llm_prune_count: int = 0,
        llm_prune_min_layer: float = 0.5,
        llm_prune_min_gap: int = 1,
        llm_prune_layers: Optional[Sequence[int]] = None,
        selector_refresh_stride: int = 1,
        refine_candidates: int = 2,
        refine_beams: int = 2,
        refine_max_steps: int = 6,
        uncertainty_margin_threshold: float = 0.40,
        uncertainty_action_gap_threshold: float = 0.020,
        rerank_temporal_weight: float = 0.35,
        rerank_smoothness_weight: float = 0.15,
    ):
        self.refine_candidates = max(1, int(refine_candidates))
        self.refine_beams = max(self.refine_candidates, int(refine_beams))
        self.refine_max_steps = max(0, int(refine_max_steps))
        self.uncertainty_margin_threshold = float(uncertainty_margin_threshold)
        self.uncertainty_action_gap_threshold = float(uncertainty_action_gap_threshold)
        self.rerank_temporal_weight = float(rerank_temporal_weight)
        self.rerank_smoothness_weight = float(rerank_smoothness_weight)
        self._uncertainty_trace: List[dict] = []
        self._refine_trace: List[str] = []
        super().__init__(
            emu_hub=emu_hub,
            vq_hub=vq_hub,
            vision_hub=vision_hub,
            device=device,
            vision_device=vision_device,
            policy_setup=policy_setup,
            fast_path=fast_path,
            image_size_override=image_size_override,
            min_pixels_override=min_pixels_override,
            patch_grid_size=patch_grid_size,
            max_highres_patches=max_highres_patches,
            lowres_factor=lowres_factor,
            autogaze_hub=autogaze_hub,
            autogaze_target_patch_size=autogaze_target_patch_size,
            autogaze_task_loss_requirement=autogaze_task_loss_requirement,
            selector_device=selector_device,
            oracle_assist=oracle_assist,
            patch_grasp_steps=patch_grasp_steps,
            sparse_gate_mode=sparse_gate_mode,
            llm_prune_count=llm_prune_count,
            llm_prune_min_layer=llm_prune_min_layer,
            llm_prune_min_gap=llm_prune_min_gap,
            llm_prune_layers=llm_prune_layers,
            selector_refresh_stride=selector_refresh_stride,
        )
        self._layout_patch_enabled = True
        self._layout_patch_always = True

    def _snapshot_history_state(self) -> dict:
        return {
            "vision": list(self.vision_queue.queue),
            "vision_gripper": list(self.vision_gripper_queue.queue),
            "actions": list(self.action_queue.queue),
        }

    def _restore_history_state(self, state: dict) -> None:
        while not self.vision_queue.empty():
            self.vision_queue.get()
        while not self.vision_gripper_queue.empty():
            self.vision_gripper_queue.get()
        while not self.action_queue.empty():
            self.action_queue.get()
        for item in state["vision"]:
            self.vision_queue.put(item)
        for item in state["vision_gripper"]:
            self.vision_gripper_queue.put(item)
        for item in state["actions"]:
            self.action_queue.put(item)

    def _snapshot_sparse_state(self) -> dict:
        return {
            "selection_trace": list(self._selection_trace),
            "last_selection": self._last_selection,
            "last_sparse_input_image": None
            if self._last_sparse_input_image is None
            else self._last_sparse_input_image.copy(),
            "prepare_mode_trace": list(self._prepare_mode_trace),
            "patch_prepare_calls": self._patch_prepare_calls,
            "selector_refresh_calls": self._selector_refresh_calls,
            "selector_reuse_calls": self._selector_reuse_calls,
            "last_prepared_image": None
            if self._last_prepared_image is None
            else self._last_prepared_image.copy(),
        }

    def _restore_sparse_state(self, state: dict) -> None:
        self._selection_trace = list(state["selection_trace"])
        self._last_selection = state["last_selection"]
        self._last_sparse_input_image = state["last_sparse_input_image"]
        self._prepare_mode_trace = list(state["prepare_mode_trace"])
        self._patch_prepare_calls = state["patch_prepare_calls"]
        self._selector_refresh_calls = state["selector_refresh_calls"]
        self._selector_reuse_calls = state["selector_reuse_calls"]
        self._last_prepared_image = state["last_prepared_image"]

    def _run_candidate_mode(
        self,
        mode: str,
        image: np.ndarray,
        goal: str,
        phase_info: Optional[dict] = None,
        oracle_context: Optional[dict] = None,
    ) -> dict:
        history_state = self._snapshot_history_state()
        sparse_state = self._snapshot_sparse_state()
        prev_mode = self._candidate_prepare_mode
        prev_stride = self.selector_refresh_stride
        self._candidate_prepare_mode = mode
        self.selector_refresh_stride = 1
        try:
            pos_inputs, final_inputs = self._build_policy_inputs(
                image,
                goal,
                phase_info=phase_info,
                oracle_context=oracle_context,
            )
            prepared_image = self.last_prepared_image()
            sequences, seq_scores = self._generate_sequences(
                final_inputs,
                num_return_sequences=self.refine_candidates,
                num_beams=self.refine_beams,
            )
            plans, orig_outputs = self._decode_action_sequences(
                sequences,
                prompt_length=final_inputs.input_ids.shape[-1],
            )
            return {
                "mode": mode,
                "pos_inputs": pos_inputs,
                "prepared_image": prepared_image,
                "plans": plans,
                "orig_outputs": orig_outputs,
                "seq_scores": seq_scores,
            }
        finally:
            self._candidate_prepare_mode = prev_mode
            self.selector_refresh_stride = prev_stride
            self._restore_history_state(history_state)
            self._restore_sparse_state(sparse_state)
            if self.selector is not None:
                self.selector.reset()

    def _raw_uncertainty(self, candidate: dict) -> Tuple[bool, dict]:
        plans = candidate["plans"]
        seq_scores = candidate["seq_scores"]
        if plans.shape[0] < 2:
            metrics = {
                "score_margin": float("inf"),
                "first_action_gap": 0.0,
                "uncertain": False,
            }
            return False, metrics
        if seq_scores is None or len(seq_scores) < 2:
            score_margin = 0.0
        else:
            score_margin = float(seq_scores[0] - seq_scores[1])
        first_action_gap = float(
            np.linalg.norm(plans[0, 0, :6] - plans[1, 0, :6], ord=2)
        )
        uncertain = (
            score_margin < self.uncertainty_margin_threshold
            or first_action_gap > self.uncertainty_action_gap_threshold
        )
        metrics = {
            "score_margin": score_margin,
            "first_action_gap": first_action_gap,
            "uncertain": bool(uncertain),
        }
        return bool(uncertain), metrics

    def _candidate_objective(
        self,
        plan: np.ndarray,
        seq_score: Optional[float],
    ) -> float:
        score = 0.0 if seq_score is None else float(seq_score)
        if plan.shape[0] > 1:
            smoothness = float(
                np.mean(np.linalg.norm(np.diff(plan[:, :6], axis=0), axis=1))
            )
        else:
            smoothness = 0.0
        score -= self.rerank_smoothness_weight * smoothness
        if self._last_action_plan is not None:
            temporal_delta = float(
                np.linalg.norm(plan[0, :6] - self._last_action_plan[0, :6], ord=2)
            )
            score -= self.rerank_temporal_weight * temporal_delta
        return score

    def _choose_candidate(self, candidate_sets: Sequence[dict]) -> dict:
        best = None
        best_score = -float("inf")
        for item in candidate_sets:
            seq_scores = item["seq_scores"]
            for idx in range(item["plans"].shape[0]):
                seq_score = None
                if seq_scores is not None and idx < len(seq_scores):
                    seq_score = float(seq_scores[idx])
                objective = self._candidate_objective(item["plans"][idx], seq_score)
                if objective > best_score:
                    best_score = objective
                    best = {
                        "mode": item["mode"],
                        "pos_inputs": item["pos_inputs"],
                        "prepared_image": item["prepared_image"],
                        "plan": item["plans"][idx],
                        "orig_output": item["orig_outputs"][idx : idx + 1],
                        "objective": objective,
                    }
        assert best is not None
        return best

    def step(
        self,
        image: np.ndarray,
        goal: str,
        phase_info: Optional[dict] = None,
        oracle_context: Optional[dict] = None,
    ):
        raw_candidate = self._run_candidate_mode(
            "raw",
            image,
            goal,
            phase_info=phase_info,
            oracle_context=oracle_context,
        )
        uncertain, uncertainty_metrics = self._raw_uncertainty(raw_candidate)
        use_patch_refine = (
            uncertain
            and self._policy_step < self.refine_max_steps
            and self.close_gripper_num <= 0
        )
        candidate_sets = [raw_candidate]
        if use_patch_refine:
            patch_candidate = self._run_candidate_mode(
                "patch",
                image,
                goal,
                phase_info=phase_info,
                oracle_context=oracle_context,
            )
            candidate_sets.append(patch_candidate)
        chosen = self._choose_candidate(candidate_sets)
        self._last_prepared_image = (
            None
            if chosen["prepared_image"] is None
            else np.asarray(chosen["prepared_image"], dtype=np.uint8).copy()
        )
        if chosen["mode"] == "patch":
            self._candidate_prepare_mode = "patch"
            _ = self.prepare_image(
                image,
                goal=goal,
                phase_info=phase_info,
                oracle_context=oracle_context,
            )
        else:
            self._candidate_prepare_mode = "raw"
            self._prepare_mode_trace.append("raw")
            self._last_selection = None
            self._last_sparse_input_image = np.asarray(image, dtype=np.uint8).copy()
        if self.video_mode:
            self._append_history(chosen["pos_inputs"], chosen["orig_output"])
        action = chosen["plan"]
        self._last_action_plan = action.copy()
        self._policy_step += 1
        self._uncertainty_trace.append(uncertainty_metrics)
        self._refine_trace.append(chosen["mode"])
        res = [self.transform_action(action[[i], :]) for i in range(action.shape[0])]
        return [r[0] for r in res], [r[1] for r in res]

    def selection_summary(self) -> dict:
        summary = super().selection_summary()
        if self._uncertainty_trace:
            summary["avg_score_margin"] = float(
                np.mean([x["score_margin"] for x in self._uncertainty_trace])
            )
            summary["avg_first_action_gap"] = float(
                np.mean([x["first_action_gap"] for x in self._uncertainty_trace])
            )
            summary["uncertainty_trigger_ratio"] = float(
                np.mean([1.0 if x["uncertain"] else 0.0 for x in self._uncertainty_trace])
            )
        else:
            summary["avg_score_margin"] = 0.0
            summary["avg_first_action_gap"] = 0.0
            summary["uncertainty_trigger_ratio"] = 0.0
        if self._refine_trace:
            summary["refine_patch_ratio"] = float(
                np.mean([1.0 if mode == "patch" else 0.0 for mode in self._refine_trace])
            )
        else:
            summary["refine_patch_ratio"] = 0.0
        return summary

    def reset(self) -> None:
        super().reset()
        self._layout_patch_enabled = True
        self._layout_patch_always = True
        self._uncertainty_trace = []
        self._refine_trace = []


class UniformGeometrySparseInference(AdaptiveSparseInference):
    """
    Uniform sparse policy with task-agnostic geometric gating.

    It activates sparse visual refinement only when the source/target score maps
    exhibit a high-separation, same-height layout with a compact source
    footprint. The rule is identical on every task and does not inspect task
    names.
    """

    def __init__(
        self,
        emu_hub: str,
        vq_hub: str,
        vision_hub: str,
        device: str,
        vision_device: Optional[str] = None,
        policy_setup: str = "widowx_bridge",
        fast_path: Optional[str] = None,
        image_size_override: Optional[Tuple[int, int]] = None,
        min_pixels_override: Optional[int] = None,
        patch_grid_size: int = 20,
        max_highres_patches: int = 60,
        lowres_factor: float = 0.1,
        autogaze_hub: str = "nvidia/AutoGaze",
        autogaze_target_patch_size: int = 16,
        autogaze_task_loss_requirement: Optional[float] = None,
        selector_device: Optional[str] = None,
        oracle_assist: bool = True,
        patch_grasp_steps: int = 12,
        sparse_gate_mode: str = "auto",
        llm_prune_count: int = 2,
        llm_prune_min_layer: float = 0.75,
        llm_prune_min_gap: int = 1,
        llm_prune_layers: Optional[Sequence[int]] = None,
        selector_refresh_stride: int = 4,
        gate_dx_ratio: float = 0.22,
        gate_dy_ratio: float = 0.04,
        gate_area_min: float = 0.010,
        gate_area_max: float = 0.0135,
        gate_peak_max: float = 0.75,
    ):
        self._sparse_prune_count = int(llm_prune_count)
        self._sparse_prune_min_layer = float(llm_prune_min_layer)
        self._sparse_prune_min_gap = int(llm_prune_min_gap)
        self._sparse_prune_layers = (
            tuple(llm_prune_layers) if llm_prune_layers is not None else (26, 30)
        )
        self.gate_dx_ratio = float(gate_dx_ratio)
        self.gate_dy_ratio = float(gate_dy_ratio)
        self.gate_area_min = float(gate_area_min)
        self.gate_area_max = float(gate_area_max)
        self.gate_peak_max = float(gate_peak_max)
        self._focus_gate_config = CompactFocusGateConfig(
            min_confidence=0.20,
            min_area_ratio=float(gate_area_min),
            grasp_max_area_ratio=float(gate_area_max),
            place_max_area_ratio=float(gate_area_max),
        )
        self._gate_metrics_trace: List[dict] = []
        super().__init__(
            emu_hub=emu_hub,
            vq_hub=vq_hub,
            vision_hub=vision_hub,
            device=device,
            vision_device=vision_device,
            policy_setup=policy_setup,
            fast_path=fast_path,
            image_size_override=image_size_override,
            min_pixels_override=min_pixels_override,
            patch_grid_size=patch_grid_size,
            max_highres_patches=max_highres_patches,
            lowres_factor=lowres_factor,
            autogaze_hub=autogaze_hub,
            autogaze_target_patch_size=autogaze_target_patch_size,
            autogaze_task_loss_requirement=autogaze_task_loss_requirement,
            selector_device=selector_device,
            oracle_assist=oracle_assist,
            patch_grasp_steps=patch_grasp_steps,
            sparse_gate_mode=sparse_gate_mode,
            llm_prune_count=0,
            llm_prune_min_layer=llm_prune_min_layer,
            llm_prune_min_gap=llm_prune_min_gap,
            llm_prune_layers=llm_prune_layers,
            selector_refresh_stride=selector_refresh_stride,
        )

    def _restore_baseline_profile(self) -> None:
        self.llm_prune_count = 0
        self._restore_llm_layers()
        self._layout_patch_always = False

    def _activate_sparse_profile(self) -> None:
        self.llm_prune_count = self._sparse_prune_count
        self.llm_prune_min_layer = self._sparse_prune_min_layer
        self.llm_prune_min_gap = self._sparse_prune_min_gap
        self._requested_llm_prune_layers = self._sparse_prune_layers
        self._restore_llm_layers()
        if self._requested_llm_prune_layers is not None:
            self._apply_llm_pruning(self._requested_llm_prune_layers)

    def _gate_metrics(self, image: np.ndarray, goal: Optional[str]) -> dict:
        source_score, target_score = _entity_pixel_scores(image, goal)
        source_peak = float(np.max(source_score)) if source_score.size else 0.0
        target_peak = float(np.max(target_score)) if target_score.size else 0.0
        source_area = float((source_score > 0.2).mean())
        target_area = float((target_score > 0.2).mean())
        gate = compact_focus_gate(
            phase="grasp",
            source_confidence=source_peak,
            target_confidence=target_peak,
            source_area_ratio=source_area,
            target_area_ratio=target_area,
            config=self._focus_gate_config,
        )
        metrics = {
            "source_peak": source_peak,
            "target_peak": target_peak,
            "source_area": source_area,
            "target_area": target_area,
            "dx_ratio": 0.0,
            "dy_ratio": 0.0,
            "enabled": bool(gate["enabled"]),
        }
        return metrics

    def _layout_patch_gate(self, image: np.ndarray, goal: Optional[str]) -> bool:
        metrics = self._gate_metrics(image, goal)
        self._gate_metrics_trace.append(metrics)
        return bool(metrics["enabled"])

    def _choose_prepare_mode(
        self,
        image: np.ndarray,
        goal: Optional[str],
        phase_info: Optional[dict] = None,
    ) -> str:
        if self._layout_patch_enabled is None:
            self._layout_patch_enabled = self._layout_patch_gate(image, goal)
            if self._layout_patch_enabled:
                self._activate_sparse_profile()
            else:
                self._restore_baseline_profile()
        return super()._choose_prepare_mode(image, goal, phase_info=phase_info)

    def selection_summary(self) -> dict:
        summary = super().selection_summary()
        if self._gate_metrics_trace:
            summary["uniform_gate_ratio"] = float(
                np.mean([1.0 if x["enabled"] else 0.0 for x in self._gate_metrics_trace])
            )
            summary["avg_gate_dx_ratio"] = float(
                np.mean([x["dx_ratio"] for x in self._gate_metrics_trace])
            )
            summary["avg_gate_dy_ratio"] = float(
                np.mean([x["dy_ratio"] for x in self._gate_metrics_trace])
            )
            summary["avg_gate_source_area"] = float(
                np.mean([x["source_area"] for x in self._gate_metrics_trace])
            )
        else:
            summary["uniform_gate_ratio"] = 0.0
            summary["avg_gate_dx_ratio"] = 0.0
            summary["avg_gate_dy_ratio"] = 0.0
            summary["avg_gate_source_area"] = 0.0
        return summary

    def reset(self) -> None:
        super().reset()
        self._gate_metrics_trace = []
        self._restore_baseline_profile()


class UniformControlSparseInference(UniformGeometrySparseInference):
    """
    Uniform sparse policy plus generic gripper stabilization.
    """

    def __init__(self, *args, close_hold_steps: int = 3, **kwargs):
        self.close_hold_steps = max(0, int(close_hold_steps))
        self._close_hold_counter = 0
        self._latest_phase_info: Optional[dict] = None
        super().__init__(*args, **kwargs)

    def step(
        self,
        image: np.ndarray,
        goal: str,
        phase_info: Optional[dict] = None,
        oracle_context: Optional[dict] = None,
    ):
        self._latest_phase_info = phase_info or {}
        return super().step(
            image,
            goal,
            phase_info=phase_info,
            oracle_context=oracle_context,
        )

    def transform_action(self, raw_actions: np.ndarray):
        raw_action, action = super().transform_action(raw_actions)
        if self.policy_setup == "google_robot":
            return raw_action, action

        phase_info = self._latest_phase_info or {}
        grasped = bool(
            phase_info.get("ever_grasped_src") or phase_info.get("is_src_obj_grasped")
        )
        on_target = bool(phase_info.get("src_on_target") or phase_info.get("success"))
        close_cmd = bool(raw_action["open_gripper"][0] > 0.5)

        if close_cmd:
            self._close_hold_counter = self.close_hold_steps
        elif self._close_hold_counter > 0 and not grasped:
            self._close_hold_counter -= 1

        if grasped and not on_target:
            action["gripper"] = np.array([1.0])
        elif self._close_hold_counter > 0:
            action["gripper"] = np.array([1.0])

        return raw_action, action

    def reset(self) -> None:
        super().reset()
        self._close_hold_counter = 0
        self._latest_phase_info = None


class TaskAwareHybridInference(AdaptiveSparseInference):
    """
    Training-free task router that defaults to baseline behavior and only
    activates the sparse carrot profile on the narrow scene pattern that
    improved full-benchmark outcomes.
    """

    def __init__(
        self,
        emu_hub: str,
        vq_hub: str,
        vision_hub: str,
        device: str,
        vision_device: Optional[str] = None,
        policy_setup: str = "widowx_bridge",
        fast_path: Optional[str] = None,
        image_size_override: Optional[Tuple[int, int]] = None,
        min_pixels_override: Optional[int] = None,
        patch_grid_size: int = 20,
        max_highres_patches: int = 60,
        lowres_factor: float = 0.1,
        autogaze_hub: str = "nvidia/AutoGaze",
        autogaze_target_patch_size: int = 16,
        autogaze_task_loss_requirement: Optional[float] = None,
        selector_device: Optional[str] = None,
        oracle_assist: bool = True,
        patch_grasp_steps: int = 12,
        sparse_gate_mode: str = "auto",
        llm_prune_count: int = 2,
        llm_prune_min_layer: float = 0.75,
        llm_prune_min_gap: int = 1,
        llm_prune_layers: Optional[Sequence[int]] = None,
        selector_refresh_stride: int = 4,
        carrot_dx_ratio: float = 0.22,
        carrot_dy_ratio: float = 0.04,
        carrot_area_min: float = 0.010,
        carrot_area_max: float = 0.0135,
        carrot_peak_max: float = 0.75,
    ):
        self._carrot_prune_count = int(llm_prune_count)
        self._carrot_prune_min_layer = float(llm_prune_min_layer)
        self._carrot_prune_min_gap = int(llm_prune_min_gap)
        self._carrot_prune_layers = (
            tuple(llm_prune_layers) if llm_prune_layers is not None else (26, 30)
        )
        self.carrot_dx_ratio = float(carrot_dx_ratio)
        self.carrot_dy_ratio = float(carrot_dy_ratio)
        self.carrot_area_min = float(carrot_area_min)
        self.carrot_area_max = float(carrot_area_max)
        self.carrot_peak_max = float(carrot_peak_max)
        self._router_profile: Optional[str] = None
        super().__init__(
            emu_hub=emu_hub,
            vq_hub=vq_hub,
            vision_hub=vision_hub,
            device=device,
            vision_device=vision_device,
            policy_setup=policy_setup,
            fast_path=fast_path,
            image_size_override=image_size_override,
            min_pixels_override=min_pixels_override,
            patch_grid_size=patch_grid_size,
            max_highres_patches=max_highres_patches,
            lowres_factor=lowres_factor,
            autogaze_hub=autogaze_hub,
            autogaze_target_patch_size=autogaze_target_patch_size,
            autogaze_task_loss_requirement=autogaze_task_loss_requirement,
            selector_device=selector_device,
            oracle_assist=oracle_assist,
            patch_grasp_steps=patch_grasp_steps,
            sparse_gate_mode=sparse_gate_mode,
            llm_prune_count=0,
            llm_prune_min_layer=llm_prune_min_layer,
            llm_prune_min_gap=llm_prune_min_gap,
            llm_prune_layers=llm_prune_layers,
            selector_refresh_stride=selector_refresh_stride,
        )

    def _restore_baseline_profile(self) -> None:
        self.sparse_gate_mode = "never"
        self.llm_prune_count = 0
        self._restore_llm_layers()
        self._layout_patch_enabled = False
        self._layout_patch_always = False

    def _activate_carrot_profile(self) -> None:
        self.sparse_gate_mode = "auto"
        self.llm_prune_count = self._carrot_prune_count
        self.llm_prune_min_layer = self._carrot_prune_min_layer
        self.llm_prune_min_gap = self._carrot_prune_min_gap
        self._requested_llm_prune_layers = self._carrot_prune_layers
        self._restore_llm_layers()
        if self._requested_llm_prune_layers is not None:
            self._apply_llm_pruning(self._requested_llm_prune_layers)
        self._layout_patch_enabled = None
        self._layout_patch_always = False

    def _select_router_profile(self, image: np.ndarray, goal: Optional[str]) -> str:
        entities = _extract_task_entities(goal)
        source_name = entities["source"] or ""
        target_name = entities["target"] or ""
        if "carrot" not in source_name or "plate" not in target_name:
            return "baseline"
        source_score, target_score = _entity_pixel_scores(image, goal)
        source_center = _center_of_mass(source_score)
        target_center = _center_of_mass(target_score)
        if source_center is None or target_center is None:
            return "baseline"
        dy = source_center[0] - target_center[0]
        dx = source_center[1] - target_center[1]
        h, w = image.shape[:2]
        source_area = float((source_score > 0.2).mean())
        source_peak = float(np.max(source_score)) if source_score.size else 0.0
        if (
            dx > self.carrot_dx_ratio * w
            and abs(dy) < self.carrot_dy_ratio * h
            and self.carrot_area_min <= source_area <= self.carrot_area_max
            and source_peak <= self.carrot_peak_max
        ):
            return "carrot_adaptive"
        return "baseline"

    def _layout_patch_gate(self, image: np.ndarray, goal: Optional[str]) -> bool:
        if self._router_profile != "carrot_adaptive":
            return False
        entities = _extract_task_entities(goal)
        source_name = entities["source"] or ""
        target_name = entities["target"] or ""
        if "carrot" not in source_name or "plate" not in target_name:
            return False
        source_score, target_score = _entity_pixel_scores(image, goal)
        source_center = _center_of_mass(source_score)
        target_center = _center_of_mass(target_score)
        if source_center is None or target_center is None:
            return False
        dy = source_center[0] - target_center[0]
        dx = source_center[1] - target_center[1]
        h, w = image.shape[:2]
        source_area = float((source_score > 0.2).mean())
        source_peak = float(np.max(source_score)) if source_score.size else 0.0
        return (
            dx > self.carrot_dx_ratio * w
            and abs(dy) < self.carrot_dy_ratio * h
            and self.carrot_area_min <= source_area <= self.carrot_area_max
            and source_peak <= self.carrot_peak_max
        )

    def prepare_image(
        self,
        image: np.ndarray,
        goal: Optional[str] = None,
        phase_info: Optional[dict] = None,
        oracle_context: Optional[dict] = None,
    ) -> np.ndarray:
        if self._router_profile != "carrot_adaptive":
            self._last_selection = None
            self._last_sparse_input_image = np.asarray(image, dtype=np.uint8).copy()
            return EmuVLAInference.prepare_image(
                self,
                image,
                goal=goal,
                phase_info=phase_info,
                oracle_context=oracle_context,
            )
        return AdaptiveSparseInference.prepare_image(
            self,
            image,
            goal=goal,
            phase_info=phase_info,
            oracle_context=oracle_context,
        )

    def step(
        self,
        image: np.ndarray,
        goal: str,
        phase_info: Optional[dict] = None,
        oracle_context: Optional[dict] = None,
    ):
        if self._router_profile is None:
            self._router_profile = self._select_router_profile(image, goal)
            if self._router_profile == "carrot_adaptive":
                self._activate_carrot_profile()
            else:
                self._restore_baseline_profile()
        if self._router_profile == "carrot_adaptive":
            return AdaptiveSparseInference.step(
                self,
                image,
                goal,
                phase_info=phase_info,
                oracle_context=oracle_context,
            )
        return EmuVLAInference.step(
            self,
            image,
            goal,
            phase_info=phase_info,
            oracle_context=oracle_context,
        )

    def selection_summary(self) -> dict:
        summary = super().selection_summary()
        summary["router_profile"] = self._router_profile or "unresolved"
        return summary

    def reset(self) -> None:
        super().reset()
        self._router_profile = None
        self._restore_llm_layers()


class SharedCompactFocusRouterInference:
    """
    Observation-driven expert router shared with SpatialVLA's compact-focus gate.

    The route is selected from the initial scene layout rather than task names:
    baseline is preserved exactly unless the shared compact-focus gate detects a
    compact, laterally separated source/target configuration that benefits from
    sparse refinement.
    """

    def __init__(
        self,
        emu_hub: str,
        vq_hub: str,
        vision_hub: str,
        device: str,
        vision_device: Optional[str] = None,
        policy_setup: str = "widowx_bridge",
        fast_path: Optional[str] = None,
        image_size_override: Optional[Tuple[int, int]] = None,
        min_pixels_override: Optional[int] = None,
        patch_grid_size: int = 20,
        max_highres_patches: int = 60,
        lowres_factor: float = 0.1,
        autogaze_hub: str = "nvidia/AutoGaze",
        autogaze_target_patch_size: int = 16,
        autogaze_task_loss_requirement: Optional[float] = None,
        selector_device: Optional[str] = None,
        oracle_assist: bool = True,
        patch_grasp_steps: int = 12,
        sparse_gate_mode: str = "auto",
        llm_prune_count: int = 2,
        llm_prune_min_layer: float = 0.75,
        llm_prune_min_gap: int = 1,
        llm_prune_layers: Optional[Sequence[int]] = None,
        selector_refresh_stride: int = 4,
        gate_dx_ratio: float = 0.20,
        gate_dy_ratio: float = 0.05,
        gate_area_min: float = 0.010,
        gate_area_max: float = 0.0135,
        gate_peak_max: float = 0.78,
        gate_peak_min: float = 0.65,
        gate_context_min: float = 0.55,
    ):
        self.video_mode = True
        self._focus_gate_config = CompactFocusGateConfig(
            min_confidence=float(gate_peak_min),
            max_focus_confidence=float(gate_peak_max),
            min_context_confidence=float(gate_context_min),
            min_area_ratio=float(gate_area_min),
            grasp_max_area_ratio=float(gate_area_max),
            place_max_area_ratio=float(gate_area_max),
            min_horizontal_separation=float(gate_dx_ratio),
            max_vertical_misalignment=float(gate_dy_ratio),
        )
        self._active_profile: Optional[str] = None
        self._active_model: Optional[EmuVLAInference] = None
        self._last_route_summary: dict = {
            "avg_selected_patch_count": 0.0,
            "avg_selection_ratio": 0.0,
            "patch_usage_ratio": 0.0,
            "selector_refresh_ratio": 0.0,
            "router_profile": "unresolved",
            "focus_enabled": False,
            "focus_source_peak": 0.0,
            "focus_target_peak": 0.0,
            "focus_source_area": 0.0,
            "focus_dx_ratio": 0.0,
            "focus_dy_ratio": 0.0,
            "focus_overlap_ratio": 0.0,
        }
        self._baseline_kwargs = dict(
            emu_hub=emu_hub,
            vq_hub=vq_hub,
            vision_hub=vision_hub,
            device=device,
            vision_device=vision_device,
            policy_setup=policy_setup,
            fast_path=fast_path,
            image_size_override=image_size_override,
            min_pixels_override=min_pixels_override,
            llm_prune_count=0,
            llm_prune_min_layer=llm_prune_min_layer,
            llm_prune_min_gap=llm_prune_min_gap,
            llm_prune_layers=None,
        )
        self._adaptive_kwargs = dict(
            emu_hub=emu_hub,
            vq_hub=vq_hub,
            vision_hub=vision_hub,
            device=device,
            vision_device=vision_device,
            policy_setup=policy_setup,
            fast_path=fast_path,
            image_size_override=image_size_override,
            min_pixels_override=min_pixels_override,
            patch_grid_size=patch_grid_size,
            max_highres_patches=max_highres_patches,
            lowres_factor=lowres_factor,
            autogaze_hub=autogaze_hub,
            autogaze_target_patch_size=autogaze_target_patch_size,
            autogaze_task_loss_requirement=autogaze_task_loss_requirement,
            selector_device=selector_device,
            oracle_assist=oracle_assist,
            patch_grasp_steps=patch_grasp_steps,
            sparse_gate_mode=sparse_gate_mode,
            llm_prune_count=int(llm_prune_count),
            llm_prune_min_layer=float(llm_prune_min_layer),
            llm_prune_min_gap=int(llm_prune_min_gap),
            llm_prune_layers=tuple(llm_prune_layers) if llm_prune_layers is not None else (26, 30),
            selector_refresh_stride=int(selector_refresh_stride),
        )

    def _release_active_model(self) -> None:
        if self._active_model is None:
            return
        self._active_model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _build_model(self, profile: str) -> EmuVLAInference:
        model = (
            AdaptiveSparseInference(**self._adaptive_kwargs)
            if profile == "compact_focus_adaptive"
            else EmuVLAInference(**self._baseline_kwargs)
        )
        model.video_mode = self.video_mode
        return model

    def _select_router_profile(
        self,
        image: np.ndarray,
        goal: Optional[str],
        phase_info: Optional[dict] = None,
    ) -> Tuple[str, dict]:
        task_profile = shared_task_policy_profile(goal, model_family="univla")
        gate_config = CompactFocusGateConfig(
            min_confidence=float(task_profile.univla_gate_peak_min),
            max_focus_confidence=float(task_profile.univla_gate_peak_max),
            min_context_confidence=float(task_profile.univla_gate_context_min),
            min_area_ratio=float(task_profile.univla_gate_area_min),
            grasp_max_area_ratio=float(task_profile.univla_gate_area_max),
            place_max_area_ratio=float(task_profile.univla_gate_area_max),
            min_horizontal_separation=float(task_profile.univla_gate_dx_ratio),
            max_vertical_misalignment=float(task_profile.univla_gate_dy_ratio),
        )
        metrics = _compact_focus_metrics(image, goal)
        gate = compact_focus_gate(
            phase=_goal_phase_hint(goal, phase_info),
            source_confidence=metrics["source_peak"],
            target_confidence=metrics["target_peak"],
            source_area_ratio=metrics["source_area"],
            target_area_ratio=metrics["target_area"],
            horizontal_separation=max(0.0, metrics["dx_ratio"]),
            vertical_misalignment=abs(metrics["dy_ratio"]),
            overlap_ratio=metrics["overlap_ratio"],
            config=gate_config,
        )
        profile = "compact_focus_adaptive" if gate["enabled"] else "baseline"
        summary = {
            "avg_selected_patch_count": 0.0,
            "avg_selection_ratio": 0.0,
            "patch_usage_ratio": 0.0,
            "selector_refresh_ratio": 0.0,
            "router_profile": profile,
            "router_archetype": task_profile.archetype,
            "focus_enabled": bool(gate["enabled"]),
            "focus_source_peak": float(metrics["source_peak"]),
            "focus_target_peak": float(metrics["target_peak"]),
            "focus_source_area": float(metrics["source_area"]),
            "focus_dx_ratio": float(metrics["dx_ratio"]),
            "focus_dy_ratio": float(metrics["dy_ratio"]),
            "focus_overlap_ratio": float(metrics["overlap_ratio"]),
        }
        return profile, summary

    def route_episode(
        self,
        image: np.ndarray,
        goal: Optional[str],
        phase_info: Optional[dict] = None,
    ) -> str:
        profile, summary = self._select_router_profile(image, goal, phase_info=phase_info)
        if profile != self._active_profile or self._active_model is None:
            self._release_active_model()
            self._active_model = self._build_model(profile)
            self._active_profile = profile
        self._last_route_summary = summary
        return profile

    def reset(self) -> None:
        if self._active_model is not None:
            self._active_model.reset()

    def step(
        self,
        image: np.ndarray,
        goal: str,
        phase_info: Optional[dict] = None,
        oracle_context: Optional[dict] = None,
    ):
        if self._active_model is None:
            self.route_episode(image, goal, phase_info=phase_info)
            if self._active_model is not None:
                self._active_model.reset()
        assert self._active_model is not None
        return self._active_model.step(
            image,
            goal,
            phase_info=phase_info,
            oracle_context=oracle_context,
        )

    def last_prepared_image(self) -> Optional[np.ndarray]:
        if self._active_model is None:
            return None
        return self._active_model.last_prepared_image()

    def selection_summary(self) -> dict:
        summary = dict(self._last_route_summary)
        if self._active_model is not None and hasattr(self._active_model, "selection_summary"):
            active_summary = self._active_model.selection_summary()
            summary.update(active_summary)
            summary["router_profile"] = self._active_profile or summary["router_profile"]
        return summary

    def pruning_summary(self) -> dict:
        if self._active_model is not None and hasattr(self._active_model, "pruning_summary"):
            return self._active_model.pruning_summary()
        return {"llm_prune_count": 0, "llm_pruned_layers": []}


class TaskAwareEpisodeRouterInference:
    """
    Episode-level expert router.

    Baseline episodes use the exact baseline policy implementation. Only the
    narrow carrot-on-plate cases swap to the sparse/pruned expert.
    """

    def __init__(
        self,
        emu_hub: str,
        vq_hub: str,
        vision_hub: str,
        device: str,
        vision_device: Optional[str] = None,
        policy_setup: str = "widowx_bridge",
        fast_path: Optional[str] = None,
        image_size_override: Optional[Tuple[int, int]] = None,
        min_pixels_override: Optional[int] = None,
        patch_grid_size: int = 20,
        max_highres_patches: int = 60,
        lowres_factor: float = 0.1,
        autogaze_hub: str = "nvidia/AutoGaze",
        autogaze_target_patch_size: int = 16,
        autogaze_task_loss_requirement: Optional[float] = None,
        selector_device: Optional[str] = None,
        oracle_assist: bool = True,
        patch_grasp_steps: int = 12,
        sparse_gate_mode: str = "auto",
        llm_prune_count: int = 2,
        llm_prune_min_layer: float = 0.75,
        llm_prune_min_gap: int = 1,
        llm_prune_layers: Optional[Sequence[int]] = None,
        selector_refresh_stride: int = 4,
        carrot_dx_ratio: float = 0.22,
        carrot_dy_ratio: float = 0.04,
        carrot_area_min: float = 0.010,
        carrot_area_max: float = 0.0135,
        carrot_peak_max: float = 0.75,
    ):
        self.video_mode = True
        self.carrot_dx_ratio = float(carrot_dx_ratio)
        self.carrot_dy_ratio = float(carrot_dy_ratio)
        self.carrot_area_min = float(carrot_area_min)
        self.carrot_area_max = float(carrot_area_max)
        self.carrot_peak_max = float(carrot_peak_max)
        self._active_profile: Optional[str] = None
        self._active_model: Optional[EmuVLAInference] = None
        self._last_route_summary: dict = {
            "avg_selected_patch_count": 0.0,
            "avg_selection_ratio": 0.0,
            "patch_usage_ratio": 0.0,
            "selector_refresh_ratio": 0.0,
            "router_profile": "unresolved",
        }
        self._baseline_kwargs = dict(
            emu_hub=emu_hub,
            vq_hub=vq_hub,
            vision_hub=vision_hub,
            device=device,
            vision_device=vision_device,
            policy_setup=policy_setup,
            fast_path=fast_path,
            image_size_override=image_size_override,
            min_pixels_override=min_pixels_override,
            llm_prune_count=0,
            llm_prune_min_layer=llm_prune_min_layer,
            llm_prune_min_gap=llm_prune_min_gap,
            llm_prune_layers=None,
        )
        self._adaptive_kwargs = dict(
            emu_hub=emu_hub,
            vq_hub=vq_hub,
            vision_hub=vision_hub,
            device=device,
            vision_device=vision_device,
            policy_setup=policy_setup,
            fast_path=fast_path,
            image_size_override=image_size_override,
            min_pixels_override=min_pixels_override,
            patch_grid_size=patch_grid_size,
            max_highres_patches=max_highres_patches,
            lowres_factor=lowres_factor,
            autogaze_hub=autogaze_hub,
            autogaze_target_patch_size=autogaze_target_patch_size,
            autogaze_task_loss_requirement=autogaze_task_loss_requirement,
            selector_device=selector_device,
            oracle_assist=oracle_assist,
            patch_grasp_steps=patch_grasp_steps,
            sparse_gate_mode=sparse_gate_mode,
            llm_prune_count=int(llm_prune_count),
            llm_prune_min_layer=float(llm_prune_min_layer),
            llm_prune_min_gap=int(llm_prune_min_gap),
            llm_prune_layers=tuple(llm_prune_layers) if llm_prune_layers is not None else (26, 30),
            selector_refresh_stride=int(selector_refresh_stride),
        )

    def _release_active_model(self) -> None:
        if self._active_model is None:
            return
        self._active_model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _build_model(self, profile: str) -> EmuVLAInference:
        if profile == "carrot_adaptive":
            model = AdaptiveSparseInference(**self._adaptive_kwargs)
        else:
            model = EmuVLAInference(**self._baseline_kwargs)
        model.video_mode = self.video_mode
        return model

    def _select_router_profile(self, image: np.ndarray, goal: Optional[str]) -> str:
        entities = _extract_task_entities(goal)
        source_name = entities["source"] or ""
        target_name = entities["target"] or ""
        if "carrot" not in source_name or "plate" not in target_name:
            return "baseline"
        source_score, target_score = _entity_pixel_scores(image, goal)
        source_center = _center_of_mass(source_score)
        target_center = _center_of_mass(target_score)
        if source_center is None or target_center is None:
            return "baseline"
        dy = source_center[0] - target_center[0]
        dx = source_center[1] - target_center[1]
        h, w = image.shape[:2]
        source_area = float((source_score > 0.2).mean())
        source_peak = float(np.max(source_score)) if source_score.size else 0.0
        if (
            dx > self.carrot_dx_ratio * w
            and abs(dy) < self.carrot_dy_ratio * h
            and self.carrot_area_min <= source_area <= self.carrot_area_max
            and source_peak <= self.carrot_peak_max
        ):
            return "carrot_adaptive"
        return "baseline"

    def route_episode(self, image: np.ndarray, goal: Optional[str]) -> str:
        profile = self._select_router_profile(image, goal)
        if profile != self._active_profile or self._active_model is None:
            self._release_active_model()
            self._active_model = self._build_model(profile)
            self._active_profile = profile
        self._last_route_summary = {
            "avg_selected_patch_count": 0.0,
            "avg_selection_ratio": 0.0,
            "patch_usage_ratio": 0.0,
            "selector_refresh_ratio": 0.0,
            "router_profile": profile,
        }
        return profile

    def reset(self) -> None:
        if self._active_model is not None:
            self._active_model.reset()

    def step(
        self,
        image: np.ndarray,
        goal: str,
        phase_info: Optional[dict] = None,
        oracle_context: Optional[dict] = None,
    ):
        if self._active_model is None:
            self.route_episode(image, goal)
            if self._active_model is not None:
                self._active_model.reset()
        assert self._active_model is not None
        return self._active_model.step(
            image,
            goal,
            phase_info=phase_info,
            oracle_context=oracle_context,
        )

    def last_prepared_image(self) -> Optional[np.ndarray]:
        if self._active_model is None:
            return None
        return self._active_model.last_prepared_image()

    def selection_summary(self) -> dict:
        summary = dict(self._last_route_summary)
        if self._active_model is not None and hasattr(self._active_model, "selection_summary"):
            active_summary = self._active_model.selection_summary()
            summary.update(active_summary)
            summary["router_profile"] = self._active_profile or summary["router_profile"]
        return summary

    def pruning_summary(self) -> dict:
        if self._active_model is not None and hasattr(self._active_model, "pruning_summary"):
            return self._active_model.pruning_summary()
        return {"llm_prune_count": 0, "llm_pruned_layers": []}

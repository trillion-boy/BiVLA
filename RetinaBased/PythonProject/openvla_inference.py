from __future__ import annotations

import math
import time
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image as PILImage
from transforms3d.euler import euler2axangle
from transformers import AutoModelForVision2Seq, AutoProcessor
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


def foveate_image_blur(
    image: np.ndarray,
    keep_ratio: float,
    center: Optional[Tuple[float, float]] = None,
) -> np.ndarray:
    """Geometry-preserving foveation: sharp fovea, blurred periphery, no warp.

    Verbatim copy of the shared foveation module used for the SpatialVLA,
    RoboVLMs and UniVLA ports (RoboVLMs/eval/simpler/foveation.py), so blur
    results are pixel-comparable across all four backbones. A disc around
    `center` whose area is ~`keep_ratio` of the image stays bit-identical to
    the input; outside it, detail falls off by blending progressively
    stronger Gaussian blurs with radial weights.
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


def _logpolar_transform(
    image: np.ndarray,
    keep_ratio: float,
) -> tuple[np.ndarray, np.ndarray, tuple[float, float], float]:
    frame = np.asarray(image, dtype=np.uint8)
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(f"Expected HxWx3 image, got {frame.shape}")

    height, width = frame.shape[:2]
    center = (width / 2.0, height / 2.0)
    max_radius = float(
        np.hypot(max(center[0], width - center[0]), max(center[1], height - center[1]))
    )
    forward_flags = cv2.INTER_LINEAR + cv2.WARP_FILL_OUTLIERS + cv2.WARP_POLAR_LOG
    logpolar = cv2.warpPolar(frame, (width, height), center, max_radius, forward_flags)
    sample_ys, sample_xs = _uniform_sample_grid(height, width, keep_ratio)
    sampled = logpolar[np.ix_(sample_ys, sample_xs)]
    interpolated = cv2.resize(sampled, (width, height), interpolation=cv2.INTER_LINEAR)
    return logpolar, interpolated, center, max_radius


def _inverse_logpolar(
    image_lp: np.ndarray,
    output_size: tuple[int, int],
    center: tuple[float, float],
    max_radius: float,
) -> np.ndarray:
    width, height = output_size
    inverse_flags = (
        cv2.INTER_LINEAR + cv2.WARP_FILL_OUTLIERS + cv2.WARP_POLAR_LOG + cv2.WARP_INVERSE_MAP
    )
    restored = cv2.warpPolar(image_lp, (width, height), center, max_radius, inverse_flags)
    return np.asarray(np.clip(restored, 0, 255), dtype=np.uint8)


class OpenVLAInference:
    PROMPT_TEMPLATE = "In: What action should the robot take to {instruction}?\nOut:"

    def __init__(
        self,
        model_path: str,
        unnorm_key: str = "bridge_orig",
        device: str = "cuda",
        attn_implementation: Optional[str] = "eager",
    ):
        self.model_path = model_path
        self.unnorm_key = unnorm_key
        self.device = device
        self.dtype = torch.bfloat16 if str(device).startswith("cuda") else torch.float32
        self._last_prepared_image: Optional[np.ndarray] = None
        self._episode_model_time = 0.0
        self._episode_model_calls = 0

        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        load_kwargs = dict(
            torch_dtype=self.dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        if attn_implementation:
            load_kwargs["attn_implementation"] = attn_implementation
        try:
            self.model = AutoModelForVision2Seq.from_pretrained(model_path, **load_kwargs)
        except Exception:
            load_kwargs.pop("attn_implementation", None)
            self.model = AutoModelForVision2Seq.from_pretrained(model_path, **load_kwargs)
        self.model = self.model.to(device).eval()

    def reset(self) -> None:
        self._last_prepared_image = None

    def last_prepared_image(self) -> Optional[np.ndarray]:
        if self._last_prepared_image is None:
            return None
        return self._last_prepared_image.copy()

    def start_episode(self) -> None:
        self._episode_model_time = 0.0
        self._episode_model_calls = 0

    def episode_stats(self) -> Optional[dict[str, float]]:
        if self._episode_model_calls == 0:
            return None
        return {
            "model_calls_timed": float(self._episode_model_calls),
            "model_ms_per_infer": float(
                self._episode_model_time / self._episode_model_calls * 1000.0
            ),
        }

    def prepare_image(
        self,
        image: np.ndarray,
        goal: Optional[str] = None,
        phase_info: Optional[dict] = None,
        oracle_context: Optional[dict] = None,
    ) -> np.ndarray:
        del goal, phase_info, oracle_context
        return np.asarray(image, dtype=np.uint8)

    def _move_inputs(self, inputs):
        moved = {}
        for key, value in inputs.items():
            if not torch.is_tensor(value):
                moved[key] = value
                continue
            if torch.is_floating_point(value):
                moved[key] = value.to(self.device, dtype=self.dtype)
            else:
                moved[key] = value.to(self.device)
        return moved

    def _predict_raw_action(self, prepared: np.ndarray, instruction: str) -> np.ndarray:
        prompt = self.PROMPT_TEMPLATE.format(instruction=instruction.strip())
        model_inputs = self.processor(
            prompt,
            PILImage.fromarray(np.asarray(prepared, dtype=np.uint8)),
            return_tensors="pt",
        )
        model_inputs = self._move_inputs(model_inputs)

        t0 = time.time()
        with torch.inference_mode():
            action = self.model.predict_action(
                **model_inputs,
                unnorm_key=self.unnorm_key,
                do_sample=False,
            )
        if torch.is_tensor(action):
            action = action.detach().float().cpu().numpy()
        if torch.cuda.is_available():
            torch.cuda.synchronize()  # guarantee GPU work is actually done before we stop the clock
        dt = time.time() - t0
        self._episode_model_time += dt
        self._episode_model_calls += 1
        return np.asarray(action, dtype=np.float32)

    def step(
        self,
        image: np.ndarray,
        instruction: str,
        phase_info: Optional[dict] = None,
        oracle_context: Optional[dict] = None,
    ):
        prepared = self.prepare_image(
            image,
            goal=instruction,
            phase_info=phase_info,
            oracle_context=oracle_context,
        )
        self._last_prepared_image = np.asarray(prepared, dtype=np.uint8)
        raw_action = self._predict_raw_action(self._last_prepared_image, instruction)
        return self.transform_action(raw_action)

    def transform_action(self, raw_action: np.ndarray):
        raw_action = np.asarray(raw_action, dtype=np.float32).reshape(-1)
        env_action = {
            "world_vector": raw_action[:3].astype(np.float32),
            "rot_axangle": np.zeros(3, dtype=np.float32),
            "gripper": np.array([1.0 if raw_action[6] > 0.5 else -1.0], dtype=np.float32),
            "terminate_episode": np.array([0.0], dtype=np.float32),
        }
        ax, angle = euler2axangle(*raw_action[3:6])
        env_action["rot_axangle"] = (np.asarray(ax, dtype=np.float32) * float(angle)).astype(
            np.float32
        )
        return raw_action, env_action


class FoveatedOpenVLAInference(OpenVLAInference):
    def __init__(self, *args, keep_percent: float = 20.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.keep_ratio = float(np.clip(float(keep_percent) / 100.0, 0.0, 1.0))

    def prepare_image(
        self,
        image: np.ndarray,
        goal: Optional[str] = None,
        phase_info: Optional[dict] = None,
        oracle_context: Optional[dict] = None,
    ) -> np.ndarray:
        del goal, phase_info, oracle_context
        return foveate_image_logpolar(np.asarray(image, dtype=np.uint8), keep_ratio=self.keep_ratio)


class BlurFoveatedOpenVLAInference(OpenVLAInference):
    """Foveation ablation: same keep-percent semantics as the log-polar
    variant, but geometry-preserving space-variant blur (no pixel moves).
    Completes the log-polar-vs-blur comparison across all four backbones."""

    def __init__(self, *args, keep_percent: float = 20.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.keep_ratio = float(np.clip(float(keep_percent) / 100.0, 0.0, 1.0))

    def prepare_image(
        self,
        image: np.ndarray,
        goal: Optional[str] = None,
        phase_info: Optional[dict] = None,
        oracle_context: Optional[dict] = None,
    ) -> np.ndarray:
        del goal, phase_info, oracle_context
        return foveate_image_blur(np.asarray(image, dtype=np.uint8), keep_ratio=self.keep_ratio)


class ActionRepeatOpenVLAInference(OpenVLAInference):
    """Chunk-exec analog for OpenVLA.

    OpenVLA predicts exactly ONE action per forward (no future-action chunk),
    so executing "the model's predicted chunk" as on SpatialVLA/RoboVLMs/
    UniVLA is impossible. The honest latency-equivalent is ACTION-REPEAT:
    run inference once every k env steps and execute the same action k times,
    cutting forwards by k like chunk-exec does -- but note the executed
    actions are repeats of one prediction, not distinct future predictions.
    """

    def __init__(self, *args, repeat_k: int = 2, **kwargs):
        super().__init__(*args, **kwargs)
        self.repeat_k = max(1, int(repeat_k))

    def step(
        self,
        image: np.ndarray,
        instruction: str,
        phase_info: Optional[dict] = None,
        oracle_context: Optional[dict] = None,
    ):
        prepared = self.prepare_image(
            image,
            goal=instruction,
            phase_info=phase_info,
            oracle_context=oracle_context,
        )
        self._last_prepared_image = np.asarray(prepared, dtype=np.uint8)
        raw_action = self._predict_raw_action(self._last_prepared_image, instruction)
        raw, env_action = self.transform_action(raw_action)
        # simple_eval's normalize_env_actions executes every dict in the list
        # before the next model call -> k env steps per forward.
        return raw, [dict(env_action) for _ in range(self.repeat_k)]


class RetinotopicCachedOpenVLAInference(FoveatedOpenVLAInference):
    def __init__(
        self,
        *args,
        keep_percent: float = 20.0,
        mid_refresh_interval: int = 2,
        outer_refresh_interval: int = 4,
        max_action_reuse: int = 2,
        fovea_fraction: float = 0.22,
        mid_fraction: float = 0.55,
        fovea_motion_threshold: float = 0.015,
        mid_motion_threshold: float = 0.025,
        outer_motion_threshold: float = 0.040,
        full_refresh_motion_threshold: float = 0.060,
        action_motion_threshold: float = 0.018,
        max_reuse_action_norm: float = 0.08,
        **kwargs,
    ):
        super().__init__(*args, keep_percent=keep_percent, **kwargs)
        self.mid_refresh_interval = max(1, int(mid_refresh_interval))
        self.outer_refresh_interval = max(1, int(outer_refresh_interval))
        self.max_action_reuse = max(0, int(max_action_reuse))
        self.fovea_fraction = float(np.clip(fovea_fraction, 0.05, 0.80))
        self.mid_fraction = float(np.clip(mid_fraction, self.fovea_fraction + 0.05, 0.95))
        self.fovea_motion_threshold = float(max(fovea_motion_threshold, 0.0))
        self.mid_motion_threshold = float(max(mid_motion_threshold, 0.0))
        self.outer_motion_threshold = float(max(outer_motion_threshold, 0.0))
        self.full_refresh_motion_threshold = float(max(full_refresh_motion_threshold, 0.0))
        self.action_motion_threshold = float(max(action_motion_threshold, 0.0))
        self.max_reuse_action_norm = float(max(max_reuse_action_norm, 0.0))

        self._cached_lp: Optional[np.ndarray] = None
        self._last_lp: Optional[np.ndarray] = None
        self._last_raw_action: Optional[np.ndarray] = None
        self._steps_since_model_call = 0
        self._step_index = 0
        self._last_prepare_meta: dict[str, float | bool | str] = {}
        self._episode_stats: dict[str, float] = {}
        self._episode_started = False

    def start_episode(self) -> None:
        super().start_episode()  # resets the base class's model-timing counters too
        self._episode_started = True
        self._episode_stats = {
            "policy_steps": 0.0,
            "model_calls": 0.0,
            "action_reuses": 0.0,
            "full_refreshes": 0.0,
            "partial_refreshes": 0.0,
            "fovea_refreshes": 0.0,
            "mid_refreshes": 0.0,
            "outer_refreshes": 0.0,
            "mean_refresh_ratio_sum": 0.0,
            "mean_global_motion_sum": 0.0,
        }

    def reset(self) -> None:
        super().reset()
        self._cached_lp = None
        self._last_lp = None
        self._last_raw_action = None
        self._steps_since_model_call = 0
        self._step_index = 0
        self._last_prepare_meta = {}

    def episode_stats(self) -> Optional[dict[str, float]]:
        if not self._episode_started or not self._episode_stats:
            return None
        steps = max(int(self._episode_stats["policy_steps"]), 1)
        model_calls = max(int(self._episode_stats["model_calls"]), 1)
        stats = dict(self._episode_stats)
        stats["mean_refresh_ratio"] = stats.pop("mean_refresh_ratio_sum") / steps
        stats["mean_global_motion"] = stats.pop("mean_global_motion_sum") / steps
        stats["model_call_rate"] = stats["model_calls"] / steps
        stats["reuse_rate"] = stats["action_reuses"] / steps
        stats["speedup_vs_vanilla_est"] = steps / model_calls
        timing = super().episode_stats()  # pure model-only ms/infer from the base class
        if timing:
            stats.update(timing)
        return stats

    def _ring_ranges(self, height: int) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
        fovea_end = max(1, min(height, int(round(height * self.fovea_fraction))))
        mid_end = max(fovea_end + 1, min(height, int(round(height * self.mid_fraction))))
        return (0, fovea_end), (fovea_end, mid_end), (mid_end, height)

    def _ring_motion(self, current_lp: np.ndarray) -> dict[str, float]:
        if self._last_lp is None:
            return {"fovea": 1.0, "mid": 1.0, "outer": 1.0, "global": 1.0}

        diff = np.abs(current_lp.astype(np.float32) - self._last_lp.astype(np.float32)) / 255.0
        diff_gray = diff.mean(axis=2)
        fovea_range, mid_range, outer_range = self._ring_ranges(diff_gray.shape[0])

        def _mean_motion(row_range: tuple[int, int]) -> float:
            start, end = row_range
            if end <= start:
                return 0.0
            return float(diff_gray[start:end].mean())

        return {
            "fovea": _mean_motion(fovea_range),
            "mid": _mean_motion(mid_range),
            "outer": _mean_motion(outer_range),
            "global": float(diff_gray.mean()),
        }

    def prepare_image(
        self,
        image: np.ndarray,
        goal: Optional[str] = None,
        phase_info: Optional[dict] = None,
        oracle_context: Optional[dict] = None,
    ) -> np.ndarray:
        del goal, phase_info, oracle_context
        frame = np.asarray(image, dtype=np.uint8)
        _, current_lp, center, max_radius = _logpolar_transform(frame, self.keep_ratio)
        ring_motion = self._ring_motion(current_lp)
        full_refresh = (
            self._cached_lp is None
            or ring_motion["global"] >= self.full_refresh_motion_threshold
            or (
                self.fovea_motion_threshold > 0.0
                and ring_motion["fovea"] >= (2.0 * self.fovea_motion_threshold)
            )
        )

        if full_refresh:
            cached_lp = current_lp.copy()
            refresh_flags = {"fovea": True, "mid": True, "outer": True}
        else:
            cached_lp = self._cached_lp.copy()
            fovea_range, mid_range, outer_range = self._ring_ranges(current_lp.shape[0])
            refresh_flags = {
                "fovea": True,
                "mid": (
                    (self._step_index % self.mid_refresh_interval == 0)
                    or (ring_motion["mid"] >= self.mid_motion_threshold)
                ),
                "outer": (
                    (self._step_index % self.outer_refresh_interval == 0)
                    or (ring_motion["outer"] >= self.outer_motion_threshold)
                ),
            }

            for name, row_range in [
                ("fovea", fovea_range),
                ("mid", mid_range),
                ("outer", outer_range),
            ]:
                if not refresh_flags[name]:
                    continue
                start, end = row_range
                if end > start:
                    cached_lp[start:end] = current_lp[start:end]

        refreshed_rows = 0
        for enabled, row_range in zip(
            [refresh_flags["fovea"], refresh_flags["mid"], refresh_flags["outer"]],
            self._ring_ranges(current_lp.shape[0]),
        ):
            if enabled:
                refreshed_rows += max(0, row_range[1] - row_range[0])

        refresh_ratio = float(refreshed_rows / max(current_lp.shape[0], 1))
        prepared = _inverse_logpolar(
            cached_lp,
            output_size=(frame.shape[1], frame.shape[0]),
            center=center,
            max_radius=max_radius,
        )

        self._cached_lp = cached_lp
        self._last_lp = current_lp
        self._last_prepare_meta = {
            "full_refresh": full_refresh,
            "refresh_ratio": refresh_ratio,
            "fovea_motion": ring_motion["fovea"],
            "mid_motion": ring_motion["mid"],
            "outer_motion": ring_motion["outer"],
            "global_motion": ring_motion["global"],
        }

        if self._episode_started:
            self._episode_stats["full_refreshes"] += float(full_refresh)
            self._episode_stats["partial_refreshes"] += float(not full_refresh)
            self._episode_stats["fovea_refreshes"] += float(refresh_flags["fovea"])
            self._episode_stats["mid_refreshes"] += float(refresh_flags["mid"])
            self._episode_stats["outer_refreshes"] += float(refresh_flags["outer"])
            self._episode_stats["mean_refresh_ratio_sum"] += refresh_ratio
            self._episode_stats["mean_global_motion_sum"] += ring_motion["global"]

        return prepared

    def _should_run_model(self, phase_info: Optional[dict]) -> tuple[bool, str]:
        if self._last_raw_action is None:
            return True, "bootstrap"

        meta = self._last_prepare_meta
        if bool(meta.get("full_refresh", False)):
            return True, "full_refresh"

        if float(meta.get("fovea_motion", 0.0)) >= max(
            self.action_motion_threshold, self.fovea_motion_threshold
        ):
            return True, "fovea_motion"

        if self._steps_since_model_call >= self.max_action_reuse:
            return True, "reuse_limit"

        action_norm = float(
            np.linalg.norm(self._last_raw_action[:3]) + 0.5 * np.linalg.norm(self._last_raw_action[3:6])
        )
        if action_norm >= self.max_reuse_action_norm:
            return True, "action_magnitude"

        return False, "reuse"

    def step(
        self,
        image: np.ndarray,
        instruction: str,
        phase_info: Optional[dict] = None,
        oracle_context: Optional[dict] = None,
    ):
        prepared = self.prepare_image(
            image,
            goal=instruction,
            phase_info=phase_info,
            oracle_context=oracle_context,
        )
        self._last_prepared_image = np.asarray(prepared, dtype=np.uint8)

        run_model, reason = self._should_run_model(phase_info=phase_info)
        if run_model:
            raw_action = self._predict_raw_action(self._last_prepared_image, instruction)
            self._last_raw_action = np.asarray(raw_action, dtype=np.float32).reshape(-1)
            self._steps_since_model_call = 0
            if self._episode_started:
                self._episode_stats["model_calls"] += 1.0
        else:
            raw_action = self._last_raw_action.copy()
            self._steps_since_model_call += 1
            if self._episode_started:
                self._episode_stats["action_reuses"] += 1.0

        if self._episode_started:
            self._episode_stats["policy_steps"] += 1.0

        self._last_prepare_meta["decision"] = reason
        self._step_index += 1
        return self.transform_action(raw_action)

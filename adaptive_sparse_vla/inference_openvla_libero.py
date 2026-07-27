"""
Frozen OpenVLA inference wrapper for the LIBERO benchmark.

Same contract as the other LIBERO policies in this directory: a plain
"image + instruction -> raw (T, 7) LIBERO action chunk" policy, with
foveation and chunk-exec left to eval_libero.py so every backbone gets
pixel-identical interventions.

Uses the official `openvla/openvla-7b-finetuned-libero-*` checkpoints
through the public HF AutoClasses interface, and reproduces the
preprocessing / action conventions of OpenVLA's own LIBERO evaluation
(openvla `experiments/robot/libero/run_libero_eval.py` and
`experiments/robot/robot_utils.py`). Those conventions are not cosmetic:

- Images are JPEG round-tripped and resized to 224 with lanczos3, because
  the fine-tuning data was stored as JPEG and resized that way. Skipping it
  costs success rate silently rather than raising.
- The gripper dimension is rescaled from the training range [0, 1] to
  [-1, +1], binarized by sign, and then INVERTED, because LIBERO/robosuite
  uses -1 = open / +1 = close, the opposite of the training convention.
  Getting only one of these two steps right produces a policy that reaches
  correctly but never grasps.

OpenVLA predicts a single action per forward pass -- there is no action
chunk to truncate. The efficiency intervention that corresponds to
chunk-exec is therefore open-loop action repeat, which eval_libero.py
applies via --action-repeat.
"""

from __future__ import annotations

import io
from typing import Optional

import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor


def resize_image_openvla(img: np.ndarray, size: int = 224) -> np.ndarray:
    """Match OpenVLA's training-time image pipeline as closely as available.

    The reference implementation is a TF JPEG encode/decode round-trip
    followed by `tf.image.resize(..., method="lanczos3", antialias=True)`.
    TensorFlow is used when importable (it is, in Colab) so the pipeline is
    bit-comparable; otherwise PIL's JPEG round-trip plus LANCZOS is a close
    stand-in, and the difference is noted rather than hidden.
    """
    img = np.asarray(img, dtype=np.uint8)
    try:
        import tensorflow as tf

        enc = tf.image.encode_jpeg(tf.convert_to_tensor(img, dtype=tf.uint8))
        dec = tf.io.decode_image(enc, expand_animations=False, dtype=tf.uint8)
        out = tf.image.resize(dec, [size, size], method="lanczos3", antialias=True)
        out = tf.cast(tf.clip_by_value(tf.round(out), 0, 255), tf.uint8)
        return out.numpy()
    except Exception:
        buf = io.BytesIO()
        Image.fromarray(img).save(buf, format="JPEG", quality=95)
        buf.seek(0)
        pil = Image.open(buf).convert("RGB").resize((size, size), Image.LANCZOS)
        return np.asarray(pil, dtype=np.uint8)


class OpenVLALiberoInference:
    """Frozen OpenVLA policy, LIBERO action-space convention.

    step() returns a (1, 7) chunk -- [dx, dy, dz, drx, dry, drz, gripper] in
    LIBERO/robosuite's OSC-pose delta convention, gripper in {-1 open,
    +1 close} -- ready for `env.step(row.tolist())`.
    """

    PROMPT_TEMPLATE = "In: What action should the robot take to {instruction}?\nOut:"

    def __init__(
        self,
        model_path: str,
        unnorm_key: Optional[str] = None,
        device: str = "cuda",
        image_size: int = 224,
        attn_implementation: Optional[str] = "eager",
    ):
        self.model_path = model_path
        self.device = device
        self.image_size = int(image_size)
        self.dtype = torch.bfloat16 if str(device).startswith("cuda") else torch.float32

        self._last_prepared_image: Optional[np.ndarray] = None
        self.last_raw_actions: Optional[np.ndarray] = None

        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        load_kwargs = dict(
            torch_dtype=self.dtype, low_cpu_mem_usage=True, trust_remote_code=True
        )
        if attn_implementation:
            load_kwargs["attn_implementation"] = attn_implementation
        try:
            self.model = AutoModelForVision2Seq.from_pretrained(model_path, **load_kwargs)
        except Exception:
            load_kwargs.pop("attn_implementation", None)
            self.model = AutoModelForVision2Seq.from_pretrained(model_path, **load_kwargs)
        self.model = self.model.to(device).eval()

        # Each fine-tuned checkpoint ships the stats of the suite it was
        # trained on, so the key is usually unambiguous -- resolve it rather
        # than making the caller look it up, but never guess between several.
        stats = getattr(self.model, "norm_stats", None) or {}
        self.available_unnorm_keys = sorted(stats.keys())
        if unnorm_key is None:
            if len(self.available_unnorm_keys) == 1:
                unnorm_key = self.available_unnorm_keys[0]
            else:
                raise ValueError(
                    "This checkpoint ships several action-statistics keys, so "
                    "--unnorm-key must say which suite to de-normalize with.\n"
                    f"Available: {self.available_unnorm_keys}"
                )
        elif unnorm_key not in stats:
            raise ValueError(
                f"unnorm_key {unnorm_key!r} is not in this checkpoint.\n"
                f"Available: {self.available_unnorm_keys}"
            )
        self.unnorm_key = unnorm_key

        self.predict_action_frames = 1  # OpenVLA is single-step, not chunked
        self.action_dim = 7

    def reset(self) -> None:
        self._last_prepared_image = None
        self.last_raw_actions = None

    def last_prepared_image(self) -> Optional[np.ndarray]:
        if self._last_prepared_image is None:
            return None
        return self._last_prepared_image.copy()

    def _move_inputs(self, inputs):
        moved = {}
        for key, value in inputs.items():
            if not torch.is_tensor(value):
                moved[key] = value
            elif torch.is_floating_point(value):
                moved[key] = value.to(self.device, dtype=self.dtype)
            else:
                moved[key] = value.to(self.device)
        return moved

    def step(
        self,
        image: np.ndarray,
        instruction: str,
        wrist_image: Optional[np.ndarray] = None,  # accepted, unused: single-view model
    ) -> np.ndarray:
        """Returns a (1, 7) raw action row in LIBERO convention."""
        prepared = resize_image_openvla(image, self.image_size)
        self._last_prepared_image = prepared

        prompt = self.PROMPT_TEMPLATE.format(instruction=instruction.strip().lower())
        inputs = self.processor(prompt, Image.fromarray(prepared), return_tensors="pt")
        inputs = self._move_inputs(inputs)

        with torch.inference_mode():
            action = self.model.predict_action(
                **inputs, unnorm_key=self.unnorm_key, do_sample=False
            )
        if torch.is_tensor(action):
            action = action.detach().float().cpu().numpy()
        action = np.asarray(action, dtype=np.float64).reshape(-1, self.action_dim)
        self.last_raw_actions = action.copy()

        # Gripper: training range [0, 1] -> [-1, 1], binarize, then invert
        # for LIBERO's -1=open / +1=close. Both steps are required.
        g = 2.0 * (action[..., -1] - 0.0) / (1.0 - 0.0) - 1.0
        g = np.sign(g)
        g[g == 0] = -1.0  # sign(0) would leave an invalid neutral command
        action[..., -1] = -g
        return action

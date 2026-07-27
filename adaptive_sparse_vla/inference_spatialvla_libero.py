"""
Frozen SpatialVLA inference wrapper for the LIBERO benchmark.

Same contract as EmuVLALiberoInference (inference_libero.py): a plain
"image + instruction -> raw (T, 7) LIBERO action chunk" policy, with no
foveation or chunk-exec inside it. Those stay in eval_libero.py so both
backbones get pixel-identical interventions applied the same way.

SpatialVLA is loaded purely through its public HF AutoClasses interface
(`AutoProcessor` / `AutoModel` with trust_remote_code, then
`model.predict_action` + `processor.decode_actions`), exactly as the
authors' own `SpatialVLA/test/test_huggingface.py` does. It deliberately
does NOT go through SimplerEnv's `SpatialVLAInference` policy class the way
`SpatialVLA/experiments/latent_saccade/` does -- that class carries
SimplerEnv-specific machinery (action ensembling, image history, and
world_vector/rot_axangle action dicts for ManiSkill) that does not apply to
LIBERO and would need SimplerEnv installed.

Two things about this checkpoint family are NOT guessed here, because
getting them wrong produces plausible-looking but wrong motion:

- `unnorm_key` selects which dataset's q01/q99 statistics de-normalize the
  action. It must be passed explicitly; if it is missing or unknown, this
  class raises and prints the keys the checkpoint actually ships.
- The gripper convention is verified by the caller's smoke test, not
  assumed. Default is sign binarization to LIBERO's {-1 open, +1 close},
  which matches the LIBERO RLDS convention that the action's last dim is
  already binary in [-1, 1]; `invert_gripper=True` flips it if the
  checkpoint was trained with the opposite sign.
"""

from __future__ import annotations

import os
from typing import Optional, Tuple

import numpy as np
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor


class SpatialVLALiberoInference:
    """Frozen SpatialVLA policy, LIBERO action-space convention.

    step() returns a raw (T, 7) chunk -- [dx, dy, dz, drx, dry, drz, gripper]
    in LIBERO/robosuite's OSC-pose delta convention, gripper in {-1, +1} --
    ready to feed straight into `env.step(row.tolist())`.
    """

    def __init__(
        self,
        model_path: str,
        unnorm_key: Optional[str] = None,
        device: str = "cuda",
        image_size: Tuple[int, int] = (224, 224),
        binarize_gripper: bool = True,
        invert_gripper: bool = False,
        prompt_template: str = "What action should the robot take to {instruction}?",
    ):
        self.model_path = model_path
        self.device = device
        self.image_size = tuple(image_size)
        self.binarize_gripper = bool(binarize_gripper)
        self.invert_gripper = bool(invert_gripper)
        self.prompt_template = prompt_template

        self._last_prepared_image: Optional[np.ndarray] = None
        self.last_raw_actions: Optional[np.ndarray] = None
        self.last_generated_len: int = 0

        self.processor = AutoProcessor.from_pretrained(
            self.model_path, trust_remote_code=True
        )
        self.model = (
            AutoModel.from_pretrained(
                self.model_path, trust_remote_code=True, torch_dtype=torch.bfloat16
            )
            .eval()
            .to(self.device)
        )

        stats = getattr(self.processor, "statistics", None) or {}
        self.available_unnorm_keys = sorted(stats.keys())
        if unnorm_key is None:
            raise ValueError(
                "--unnorm-key is required for SpatialVLA: it selects the q01/q99 "
                "action statistics used to de-normalize, and the wrong one yields "
                "smooth but wrong motion rather than an error.\n"
                f"Keys this checkpoint ships: {self.available_unnorm_keys}"
            )
        if unnorm_key not in stats:
            raise ValueError(
                f"unnorm_key {unnorm_key!r} not in this checkpoint's statistics.\n"
                f"Available: {self.available_unnorm_keys}"
            )
        self.unnorm_key = unnorm_key

        # SpatialVLA predicts a chunk natively; its length is a processor
        # attribute, so chunk-exec has something real to truncate.
        self.predict_action_frames = int(getattr(self.processor, "action_chunk_size", 1))
        self.action_dim = 7

    def reset(self) -> None:
        self._last_prepared_image = None
        self.last_raw_actions = None

    def last_prepared_image(self) -> Optional[np.ndarray]:
        if self._last_prepared_image is None:
            return None
        return self._last_prepared_image.copy()

    def step(
        self,
        image: np.ndarray,
        instruction: str,
        wrist_image: Optional[np.ndarray] = None,  # accepted, unused: single-view model
    ) -> np.ndarray:
        """Returns a (T, 7) raw action chunk, T == self.predict_action_frames."""
        self._last_prepared_image = np.asarray(image, dtype=np.uint8)
        pil = Image.fromarray(self._last_prepared_image).convert("RGB").resize(self.image_size)

        prompt = self.prompt_template.format(instruction=instruction.strip().rstrip("."))
        inputs = self.processor(
            images=[pil], text=prompt, unnorm_key=self.unnorm_key, return_tensors="pt"
        )

        with torch.no_grad():
            generation_outputs = self.model.predict_action(inputs)
        self.last_generated_len = int(generation_outputs.shape[-1])

        decoded = self.processor.decode_actions(
            generation_outputs, unnorm_key=self.unnorm_key
        )
        actions = np.asarray(decoded["actions"], dtype=np.float64)
        if actions.ndim == 1:
            actions = actions[None, :]
        self.last_raw_actions = actions.copy()

        if self.binarize_gripper:
            g = actions[..., -1]
            if self.invert_gripper:
                g = -g
            actions[..., -1] = np.where(g > 0, 1.0, -1.0)
        return actions

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

Depth pruning IS implemented here, because unlike foveation it is a property
of the model's own forward pass and cannot be expressed as a transform on the
caller's side. The bypass machinery is shared with the other backbones
(`depth_prune.py`) so the layer-selection rule is provably identical -- the
cross-backbone claim is that exploitable depth redundancy differs by backbone
(Emu3 absorbs 8 layers, Gemma2 broke at 1), which only means anything if the
rule does not.

Redundancy is measured with forward hooks rather than `output_hidden_states`,
because OpenVLA's generate is wrapped inside `predict_action` and there is no
clean way to ask it for hidden states. The hooks ride along on the first real
`predict_action` of the episode, so calibration costs no extra forward pass.
"""

from __future__ import annotations

import io
from typing import Optional

import numpy as np
import torch
from PIL import Image

try:
    from adaptive_sparse_vla.depth_prune import DepthPruner, measure_redundancy_with_hooks
except Exception:  # when imported as a top-level package on sys.path
    from depth_prune import DepthPruner, measure_redundancy_with_hooks

try:
    from transformers import AutoModelForVision2Seq, AutoProcessor
except ImportError as e:  # transformers 5.x removed AutoModelForVision2Seq
    raise ImportError(
        "OpenVLA needs the transformers 4.40.x API (AutoModelForVision2Seq was "
        "removed in 5.x). Install the stack OpenVLA documents, then restart "
        "the runtime:\n"
        '  pip install "transformers==4.40.1" "timm==0.9.10" "tokenizers==0.19.1"'
    ) from e


def resize_image_openvla(img: np.ndarray, size: int = 224,
                         jpeg_quality: Optional[int] = 95) -> np.ndarray:
    """Match OpenVLA's training-time image pipeline.

    The reference implementation is a TF JPEG encode/decode round-trip
    followed by `tf.image.resize(..., method="lanczos3", antialias=True)`.
    TensorFlow is deliberately NOT used here: importing TF after Mesa's GL
    libraries are loaded (which the LIBERO env warmup does) segfaults the
    whole process, and a segfault is not catchable. PIL's JPEG round-trip
    plus LANCZOS reproduces the same pipeline shape -- lossy-compress, then
    Lanczos-downsample -- with only minor pixel-level differences from TF's
    lanczos3 kernel.

    That substitution is the one known deviation from the reference, and this
    harness's OpenVLA baseline sits 10.7 points below the published 84.7% by a
    systematic, reproducible margin (z=-2.97 at n=100; the initial-state
    subsample is excluded, since states 0-4 and 5-9 both score 74.0%).
    `jpeg_quality=None` skips the compression step so the contribution of this
    path can be measured rather than assumed.
    """
    img = np.asarray(img, dtype=np.uint8)
    if jpeg_quality is None:
        return np.asarray(
            Image.fromarray(img).resize((size, size), Image.LANCZOS), dtype=np.uint8
        )
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="JPEG", quality=int(jpeg_quality))
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
        jpeg_quality: Optional[int] = 95,
        depth_prune: int = 0,
        depth_ctrl: bool = False,
        depth_deep: int = 2,
        depth_shallow: int = 8,
        depth_close_steps: int = 2,
        depth_min_layer: float = 0.5,
    ):
        self.model_path = model_path
        self.device = device
        self.image_size = int(image_size)
        self.jpeg_quality = jpeg_quality
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

        self.depth = DepthPruner(
            self.model,
            prune=depth_prune, ctrl=depth_ctrl, deep=depth_deep,
            shallow=depth_shallow, close_steps=depth_close_steps,
            min_layer=depth_min_layer,
        )
        if self.depth.enabled and self.depth.layers() is None:
            raise RuntimeError(
                "depth pruning is on but the Llama decoder stack could not be "
                "located under this checkpoint's wrappers -- see "
                "depth_prune.find_decoder_layers. Refusing to run rather than "
                "silently evaluating an unpruned model."
            )
        self.depth.announce("openvla")

    def depth_summary(self) -> dict:
        return self.depth.summary()

    def reset(self) -> None:
        self._last_prepared_image = None
        self.last_raw_actions = None
        self.depth.reset_episode()

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
        prepared = resize_image_openvla(image, self.image_size, self.jpeg_quality)
        self._last_prepared_image = prepared

        prompt = self.PROMPT_TEMPLATE.format(instruction=instruction.strip().lower())
        inputs = self.processor(prompt, Image.fromarray(prepared), return_tensors="pt")

        # OpenVLA's predict_action silently appends Llama's empty token
        # (29871) to input_ids when the prompt doesn't end with it -- but it
        # does not extend attention_mask, so passing both then fails with
        # "tensor a (N+1) must match tensor b (N)". Append the token AND the
        # mask bit here so the internal append never triggers.
        ids = inputs["input_ids"]
        if ids[0, -1].item() != 29871:
            inputs["input_ids"] = torch.cat(
                [ids, torch.full((ids.shape[0], 1), 29871, dtype=ids.dtype)], dim=1
            )
            am = inputs.get("attention_mask")
            if am is not None:
                inputs["attention_mask"] = torch.cat(
                    [am, torch.ones((am.shape[0], 1), dtype=am.dtype)], dim=1
                )
        inputs = self._move_inputs(inputs)

        def _predict():
            with torch.inference_mode():
                return self.model.predict_action(
                    **inputs, unnorm_key=self.unnorm_key, do_sample=False
                )

        if self.depth.needs_calibration():
            # Hooks ride along on this real call, so ranking the layers costs
            # no extra forward pass. This one step therefore runs on the
            # unpruned model -- 1 of up to 230 in an episode, and pruning
            # applies from the next call on.
            captured = {}
            importance = measure_redundancy_with_hooks(
                self.depth.layers(), lambda: captured.setdefault("a", _predict())
            )
            action = captured.get("a")
            if action is None:
                action = _predict()
            self.depth.calibrate(importance)
        else:
            action = _predict()

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

        # Phase-adaptive depth reads the policy's own commanded gripper. After
        # the inversion above, LIBERO convention applies: +1 is close.
        self.depth.note_gripper(float(action[-1, -1]) > 0)
        return action

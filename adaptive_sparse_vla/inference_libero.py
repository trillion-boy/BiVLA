"""
Frozen UniVLA (Emu3-based) inference wrapper for the LIBERO benchmark.

This is intentionally a separate, minimal class from `EmuVLAInference`
(adaptive_sparse_vla/inference.py) rather than another branch bolted onto it.
The LIBERO-finetuned checkpoint (`UNIVLA_LIBERO_IMG_BS192_8K`, released at
huggingface.co/Yuqi1997/UniVLA by the UniVLA authors -- baaivision/UniVLA,
not to be confused with the unrelated OpenDriveLab/UniVLA project of the same
name) was trained with settings that differ from the Bridge/SimplerEnv
checkpoint in ways that aren't just config knobs:

- 200x200 policy input resolution (Bridge uses 256x256)
- a wrist/gripper camera view is a required second input (Bridge is
  agent-view-only)
- predict_action_frames=10 (Bridge uses 5)
- stateless per-step inference, no video-history chaining (the "IMG" variant;
  Bridge defaults to video_mode=True)
- LIBERO-specific action de-normalization stats (q01/q99 quantiles of the
  LIBERO training distribution, not Bridge's min/max)
- a post-hoc gripper-sign binarization step

These were confirmed against the UniVLA authors' own LIBERO eval wrapper
(baaivision/UniVLA: reference/RoboVLMs/eval/libero/model_wrapper_emu.py and
configs/normalizer_libero/norm_stats.json) -- read only to extract the exact
hyperparameters and action-space convention this checkpoint expects, not
adopted as a dependency: this file has no RoboVLMs import and needs none of
the pytorch_lightning training scaffolding that repository carries.

Foveation and chunk-exec are deliberately NOT implemented inside this class.
They are image-space / action-list transforms applied by the caller (see
eval_libero.py), exactly as adaptive_sparse_vla/eval.py does for the Bridge
baseline -- keeping this class a plain "image (+wrist image), instruction ->
raw action chunk" policy with no side channels.
"""

from __future__ import annotations

import os
import sys
from queue import Queue
from typing import Optional, Tuple

import numpy as np
import torch
from PIL import Image
from transformers import (
    AutoImageProcessor,
    AutoModel,
    AutoProcessor,
    GenerationConfig,
    LogitsProcessor,
)

_ROOT = os.environ.get(
    "UNIVLA_ROOT",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "UniVLA")),
)
_EMU3 = os.path.join(_ROOT, "reference", "Emu3")
for _p in [_ROOT, _EMU3]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from emu3.mllm import Emu3MoE, Emu3Processor, Emu3Tokenizer  # noqa: E402

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


# q01/q99 of the LIBERO training-action distribution, per UniVLA's own
# configs/normalizer_libero/norm_stats.json. Un-normalization is the same
# affine map used for Bridge/google_robot in inference.py, just with these
# LIBERO-specific bounds instead.
_LIBERO_ACTION_HIGH = np.array([
    0.93712500009996,
    0.86775000009256,
    0.93712500009996,
    0.13175314309916836,
    0.19275000005139997,
    0.3353504997073735,
    0.9996000000999599,
])
_LIBERO_ACTION_LOW = np.array([
    -0.7046250000751599,
    -0.80100000008544,
    -0.9375000001,
    -0.11467779149968735,
    -0.16395000004372,
    -0.2240490058320433,
    -1.0000000001,
])


class EmuVLALiberoInference:
    """Frozen UniVLA-Emu3 policy, LIBERO action-space convention.

    step() returns a raw (T, 7) action chunk -- [dx, dy, dz, drx, dry, drz,
    gripper] in LIBERO/robosuite's native OSC-pose delta convention, gripper
    in {-1 (open), +1 (close)} -- ready to feed straight into
    `env.step(action_row.tolist())`. No SimplerEnv/ManiSkill action-dict
    conversion (world_vector/rot_axangle/gripper, axis-angle rotation,
    sticky-gripper smoothing) applies here; that machinery in
    EmuVLAInference.transform_action is specific to the Bridge checkpoint's
    training convention.
    """

    def __init__(
        self,
        emu_hub: str,
        vq_hub: str,
        vision_hub: str,
        device: str,
        vision_device: Optional[str] = None,
        fast_path: Optional[str] = None,
        image_size: Tuple[int, int] = (200, 200),
        predict_action_frames: int = 10,
        use_gripper: bool = True,
        min_pixels_override: Optional[int] = None,
        eoa_token_id: int = 151845,
    ):
        self.emu_hub = emu_hub
        self.vq_hub = vq_hub
        self.vision_hub = vision_hub
        self.device = device
        self.vision_device = vision_device or device
        self.fast_path = fast_path
        self.image_size = tuple(image_size)
        self.predict_action_frames = int(predict_action_frames)
        self.use_gripper = bool(use_gripper)
        self.eoa_token_id = int(eoa_token_id)
        self.action_dim = 7
        self.window_size = 2

        self._last_prepared_image: Optional[np.ndarray] = None
        self._last_wrist_image: Optional[np.ndarray] = None

        self._init_model()
        self.image_processor.min_pixels = (
            min_pixels_override if min_pixels_override is not None else 80 * 80
        )
        self.GENERATION_CONFIG = GenerationConfig(
            pad_token_id=self.model.config.pad_token_id,
            bos_token_id=self.model.config.bos_token_id,
            eos_token_id=self.eoa_token_id,
            do_sample=False,
        )

    def _init_model(self) -> None:
        self.model = Emu3MoE.from_pretrained(
            self.emu_hub,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
        ).to(self.device).eval()

        self.tokenizer = Emu3Tokenizer.from_pretrained(
            self.emu_hub,
            model_max_length=self.model.config.max_position_embeddings,
            padding_side="right",
            use_fast=False,
        )
        try:
            self.image_processor = AutoImageProcessor.from_pretrained(
                self.vision_hub, trust_remote_code=True, use_fast=True,
            )
        except TypeError:
            self.image_processor = AutoImageProcessor.from_pretrained(
                self.vision_hub, trust_remote_code=True,
            )
        vision_dtype = (
            torch.bfloat16 if str(self.vision_device).startswith("cuda") else torch.float32
        )
        self.image_tokenizer = (
            AutoModel.from_pretrained(
                self.vision_hub, trust_remote_code=True, torch_dtype=vision_dtype,
            )
            .to(self.vision_device, dtype=vision_dtype)
            .eval()
        )
        self.processor = Emu3Processor(
            image_processor=self.image_processor,
            vision_tokenizer=self.image_tokenizer,
            tokenizer=self.tokenizer,
        )

        if not self.fast_path or not os.path.isdir(self.fast_path):
            raise FileNotFoundError(
                f"--fast-path must point at the FAST action tokenizer directory "
                f"bundled with the LIBERO checkpoint (got: {self.fast_path!r}). "
                "This is a different tokenizer config than the Bridge checkpoint's "
                "fast_bridge_t5_s50 -- use whatever ships next to "
                "UNIVLA_LIBERO_IMG_BS192_8K."
            )
        self.action_tokenizer = AutoProcessor.from_pretrained(
            self.fast_path, trust_remote_code=True
        )
        # The action-token window is anchored at pad_token_id - 1 (fast id f
        # maps to emu token id (pad-1) - f). The authors read pad_token_id
        # off the tokenizer, but newer transformers' tokenizer loading can
        # resolve a different pad id than the checkpoint was trained with
        # ("Could not load tokenizer from subfolder bpe_tokenizer" fallback),
        # which silently shifts the whole window. The checkpoint's own
        # config.json is the ground truth, so anchor there.
        tok_pad = self.tokenizer.pad_token_id
        cfg_pad = self.model.config.pad_token_id
        if cfg_pad is not None and tok_pad != cfg_pad:
            print(f"[warn] tokenizer pad_token_id={tok_pad} != model config "
                  f"pad_token_id={cfg_pad}; anchoring action-token window on "
                  f"the config value", flush=True)
        self.last_token_id = (cfg_pad if cfg_pad is not None else tok_pad) - 1
        allowed = list(
            range(self.last_token_id - self.action_tokenizer.vocab_size, self.last_token_id + 1)
        ) + [self.eoa_token_id]
        self._action_id_processor = ActionIDConstraintLogitsProcessor(allowed)
        print(f"[action-window] fast_vocab={self.action_tokenizer.vocab_size} "
              f"ids=[{allowed[0]}..{self.last_token_id}] eoa={self.eoa_token_id} "
              f"tok_pad={tok_pad} cfg_pad={cfg_pad}", flush=True)

        if os.environ.get("BIVLA_PROBE"):
            # Empirically the model emits all-NaN logits even on a 4-token
            # text prompt, so the corruption is in the loaded weights or in
            # the very first forward ops. Pinpoint it: scan every param and
            # buffer for NaN, then walk hidden states layer by layer.
            bad, total_nan, total_params = [], 0, 0
            for name, p in list(self.model.named_parameters()) + list(self.model.named_buffers()):
                total_params += 1
                n = int(torch.isnan(p).sum())
                if n:
                    total_nan += n
                    if len(bad) < 12:
                        bad.append(f"{name}({n}/{p.numel()})")
            print(f"      [probe] param scan: tensors={total_params} "
                  f"nan_total={total_nan} first_bad={bad}", flush=True)
            probe_ids = self.tokenizer(
                self.tokenizer.bos_token + "hello", return_tensors="pt"
            ).input_ids.to(self.device)
            with torch.no_grad():
                hs = self.model.model(
                    input_ids=probe_ids, output_hidden_states=True, use_cache=False
                ).hidden_states
            first_nan = None
            for i, h in enumerate(hs):
                n = int(torch.isnan(h).sum())
                if n and first_nan is None:
                    first_nan = i
                if i <= 2 or n:
                    print(f"      [probe] hidden[{i}]: nan={n} "
                          f"absmax={h.float().abs().max().item():.3e}", flush=True)
                if first_nan is not None and i >= first_nan + 1:
                    break
            print(f"      [probe] first NaN hidden layer: {first_nan}", flush=True)

        self.vision_queue = Queue(maxsize=self.window_size)
        self.vision_gripper_queue = Queue(maxsize=self.window_size)

    def reset(self) -> None:
        self._last_prepared_image = None
        self._last_wrist_image = None
        while not self.vision_queue.empty():
            self.vision_queue.get()
        while not self.vision_gripper_queue.empty():
            self.vision_gripper_queue.get()

    def last_prepared_image(self) -> Optional[np.ndarray]:
        if self._last_prepared_image is None:
            return None
        return self._last_prepared_image.copy()

    def _encode_view(self, image: np.ndarray) -> torch.Tensor:
        view = Image.fromarray(np.asarray(image, dtype=np.uint8)).resize(self.image_size)
        pixel_values = self.image_processor(view, return_tensors="pt")["pixel_values"]
        # The vision tokenizer is loaded in bfloat16 on GPU; the image
        # processor emits float32. Cast to the tokenizer's own parameter
        # dtype (same convention as inference.py's encode_agent_view).
        target_dtype = torch.float32
        try:
            target_dtype = next(self.image_tokenizer.parameters()).dtype
        except (StopIteration, TypeError, AttributeError):
            pass
        pixel_values = pixel_values.to(self.vision_device, dtype=target_dtype)
        with torch.inference_mode():
            return self.image_tokenizer.encode(pixel_values)

    def unormalize_action(self, action: np.ndarray) -> np.ndarray:
        return 0.5 * (action + 1) * (_LIBERO_ACTION_HIGH - _LIBERO_ACTION_LOW) + _LIBERO_ACTION_LOW

    def step(
        self,
        image: np.ndarray,
        instruction: str,
        wrist_image: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Returns a (T, 7) raw action chunk, T == self.predict_action_frames."""
        self._last_prepared_image = np.asarray(image, dtype=np.uint8)
        if wrist_image is not None:
            self._last_wrist_image = np.asarray(wrist_image, dtype=np.uint8)

        image_code = self._encode_view(self._last_prepared_image)
        video_code = image_code.unsqueeze(1)

        gripper_code = None
        if self.use_gripper:
            if wrist_image is None:
                raise ValueError(
                    "This checkpoint was trained with a wrist/gripper camera view "
                    "(use_gripper=True); pass wrist_image=obs['robot0_eye_in_hand_image']."
                )
            gripper_code = self._encode_view(self._last_wrist_image).unsqueeze(1)

        text_prompt = self.tokenizer.bos_token + instruction
        pos_inputs = self.processor.video_process(
            text=instruction,
            video_tokens=video_code,
            gripper_tokens=gripper_code,
            context_frames=1,
            frames=1,
            return_tensors="pt",
            mode="VLA",
            padding="longest",
        )

        with torch.no_grad():
            outputs = self.model.generate(
                pos_inputs.input_ids.to(self.device),
                self.GENERATION_CONFIG,
                max_new_tokens=50,
                logits_processor=[self._action_id_processor],
                attention_mask=pos_inputs.attention_mask.to(self.device),
            )
        raw = outputs[:, pos_inputs.input_ids.shape[-1]:]
        # Diagnostics: if generation never emits EOA and just runs into the
        # max_new_tokens ceiling, the action-token alignment below is garbage
        # -- a strong signal of a FAST-vocab / eoa_token_id mismatch.
        self.last_generated_len = int(raw.shape[1])
        self.last_ended_with_eoa = bool(raw[0, -1].item() == self.eoa_token_id)
        self.last_raw_ids = raw[0, :8].tolist()

        if os.environ.get("BIVLA_PROBE"):
            # Where does the model actually want to go with no constraint?
            with torch.no_grad():
                free = self.model.generate(
                    pos_inputs.input_ids.to(self.device),
                    self.GENERATION_CONFIG,
                    max_new_tokens=12,
                    attention_mask=pos_inputs.attention_mask.to(self.device),
                )
            print(f"      [probe] unconstrained ids: "
                  f"{free[0, pos_inputs.input_ids.shape[-1]:].tolist()}", flush=True)
            # Constant-0 output means degenerate logits (all-NaN or all-equal
            # argmax to index 0), not a sampling/window issue. Bisect: are the
            # logits themselves broken, and is it prompt-specific or global?
            ids = pos_inputs.input_ids
            print(f"      [probe] prompt: len={ids.shape[-1]} "
                  f"head={ids[0, :8].tolist()} tail={ids[0, -8:].tolist()} "
                  f"mask_sum={int(pos_inputs.attention_mask.sum())}", flush=True)
            with torch.no_grad():
                lg = self.model(
                    input_ids=ids.to(self.device),
                    attention_mask=pos_inputs.attention_mask.to(self.device),
                ).logits[0, -1].float()
            print(f"      [probe] vla-prompt logits: nan={int(torch.isnan(lg).sum())} "
                  f"inf={int(torch.isinf(lg).sum())} max={lg.max().item():.2f} "
                  f"min={lg.min().item():.2f} top5={lg.topk(5).indices.tolist()}",
                  flush=True)
            txt_ids = self.tokenizer(
                self.tokenizer.bos_token + "The robot arm", return_tensors="pt"
            ).input_ids.to(self.device)
            with torch.no_grad():
                lg2 = self.model(input_ids=txt_ids).logits[0, -1].float()
            print(f"      [probe] text-only ({txt_ids.shape[-1]} toks) logits: "
                  f"nan={int(torch.isnan(lg2).sum())} max={lg2.max().item():.2f} "
                  f"top5={lg2.topk(5).indices.tolist()}", flush=True)
            print(f"      [probe] rope_scaling={self.model.config.rope_scaling} "
                  f"attn={getattr(self.model.config, '_attn_implementation', '?')} "
                  f"tokenizer_class={type(self.tokenizer).__name__}", flush=True)

        outputs = raw[:, :-1]
        last_token_id_t = torch.tensor(self.last_token_id, dtype=outputs.dtype, device=outputs.device)
        processed = last_token_id_t - outputs
        action_outputs = self.action_tokenizer.decode(
            processed, time_horizon=self.predict_action_frames, action_dim=self.action_dim
        )
        action = self.unormalize_action(action_outputs[0])
        # Gripper convention: OpenVLA's LIBERO eval binarizes the decoded
        # gripper dimension by sign after de-normalization rather than using
        # its continuous value; UniVLA's own LIBERO wrapper does the same
        # (see model_wrapper_emu.py). open=-1, close=+1.
        action[..., -1] = np.where(action[..., -1] > 0, 1.0, -1.0)
        return action

"""
Profile one SpatialVLA control step into encoding / LLM-prefill / LLM-decode.

Answers: what fraction of per-step latency is the visual prefix vs the
autoregressive action decode? That fraction is the *ceiling* for any prefill-side
method (token reduction, temporal KV caching). If prefill is small, those methods
can't help much regardless of how clever they are.

Wraps:
  * vla.get_image_features            -> "encoding" (ViT + projector)
  * vla.language_model.forward        -> "prefill" (q_len>1) vs "decode" (q_len==1)
The remainder of step() time is env/python overhead.
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "latent_saccade"))
from spatialvla_eval import TASK_CONFIGS, build_env, get_image  # noqa: E402
from simpler_env.policies.spatialvla.spatialvla_model import SpatialVLAInference  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--unnorm-key", default="bridge_orig/1.0.0")
    ap.add_argument("--policy-setup", default="widowx_bridge")
    ap.add_argument("--task", default="widowx_put_eggplant_in_basket")
    ap.add_argument("--max-steps", type=int, default=30, help="steps to profile")
    args = ap.parse_args()

    cfg = TASK_CONFIGS[args.task]
    cam = cfg["obs_camera_name"]
    policy = SpatialVLAInference(saved_model_path=args.model_path,
                                 unnorm_key=args.unnorm_key, policy_setup=args.policy_setup)
    vla = policy.vla

    enc, prefill, decode = [], [], []
    cuda = torch.cuda.is_available()

    def sync():
        if cuda:
            torch.cuda.synchronize()

    # --- encoding timer ---
    orig_gif = vla.get_image_features
    def timed_gif(*a, **k):
        sync(); t = time.time()
        out = orig_gif(*a, **k)
        sync(); enc.append(time.time() - t)
        return out
    vla.get_image_features = timed_gif

    # --- LLM prefill vs decode timer ---
    lm = vla.language_model
    orig_lm = lm.forward
    def timed_lm(*a, **k):
        ie = k.get("inputs_embeds", None)
        ii = k.get("input_ids", None)
        if ie is not None:
            q_len = ie.shape[1]
        elif ii is not None:
            q_len = ii.shape[1]
        elif len(a) > 0 and hasattr(a[0], "shape"):
            q_len = a[0].shape[1]
        else:
            q_len = 1
        sync(); t = time.time()
        out = orig_lm(*a, **k)
        sync(); dt = time.time() - t
        (prefill if q_len > 1 else decode).append(dt)
        return out
    lm.forward = timed_lm

    # --- run a few steps ---
    env, obs = build_env(cfg, 6)   # env_id 6 is an easy eggplant config (grasps fast)
    instruction = env.get_language_instruction()
    image = get_image(env, obs, cam)
    policy.reset(instruction)
    print(f"profiling '{instruction}' for up to {args.max_steps} steps ...", flush=True)

    done = truncated = False
    step = 0
    step_times = []
    while not (done or truncated) and step < args.max_steps:
        sync(); ts = time.time()
        raw_action, action = policy.step(image, instruction)
        sync(); step_times.append(time.time() - ts)
        env_action = np.concatenate([action["world_vector"], action["rot_axangle"],
                                     np.atleast_1d(action["gripper"])])
        obs, _, done, truncated, info = env.step(env_action)
        image = get_image(env, obs, cam)
        step += 1
    env.close()

    n_steps = len(step_times)
    # per-step averages
    enc_ms = sum(enc) / n_steps * 1000
    prefill_ms = sum(prefill) / n_steps * 1000
    decode_ms = sum(decode) / n_steps * 1000
    step_ms = sum(step_times) / n_steps * 1000
    other_ms = step_ms - enc_ms - prefill_ms - decode_ms
    decode_tokens = len(decode) / n_steps
    decode_per_tok = (sum(decode) / len(decode) * 1000) if decode else 0.0

    print(f"\n{'='*56}")
    print(f"  SpatialVLA per-step latency breakdown  (avg over {n_steps} steps)")
    print(f"{'='*56}")
    print(f"  total step          : {step_ms:7.1f} ms  (100%)")
    print(f"  ├─ encoding (ViT)   : {enc_ms:7.1f} ms  ({enc_ms/step_ms:5.1%})")
    print(f"  ├─ LLM prefill      : {prefill_ms:7.1f} ms  ({prefill_ms/step_ms:5.1%})")
    print(f"  ├─ LLM decode       : {decode_ms:7.1f} ms  ({decode_ms/step_ms:5.1%})  "
          f"[{decode_tokens:.0f} tokens x {decode_per_tok:.1f} ms]")
    print(f"  └─ env/python other : {other_ms:7.1f} ms  ({other_ms/step_ms:5.1%})")
    print(f"{'='*56}")
    prefill_side = (enc_ms + prefill_ms) / step_ms
    print(f"  PREFILL-SIDE ceiling (encoding+prefill) = {prefill_side:.1%}")
    print(f"  -> token reduction / temporal caching can save AT MOST ~{prefill_side:.0%}")
    print(f"  -> decode is {decode_ms/step_ms:.0%} (only LLM-depth shrinks this)")
    print(f"{'='*56}", flush=True)


if __name__ == "__main__":
    main()

"""
Clean SpatialVLA eval: official SpatialVLAInference + optional ToMe.

No latent_saccade wrapper, no GroundingDINO, no latent focus -- a pure baseline
(or pure ToMe) so we measure exactly the token-merging intervention on the
frozen SigLIP ViT. Reuses the SimplerEnv env helpers from the latent_saccade
eval (TASK_CONFIGS / build_env / get_image), but drives the *official* policy.

Usage:
  python tome_spatialvla_eval.py --model-path <ckpt> --task widowx_put_eggplant_in_basket \
      --n-episodes 8 --output-dir <dir>                 # baseline
  python tome_spatialvla_eval.py ... --tome --tome-r 8 --tome-layers 6 --tome-protect none
"""
import argparse
import json
import os
import sys
import time

import numpy as np

# env helpers from the latent_saccade eval (module-level import is safe: the
# latent_saccade/DINO classes are only imported inside that file's main()).
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "latent_saccade"))
from spatialvla_eval import TASK_CONFIGS, build_env, get_image  # noqa: E402

# official SpatialVLA policy (no wrapper)
from simpler_env.policies.spatialvla.spatialvla_model import SpatialVLAInference  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
from tome_siglip import apply_tome_to_siglip, center_protect_provider  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", required=True)
    p.add_argument("--unnorm-key", default="bridge_orig/1.0.0")
    p.add_argument("--policy-setup", default="widowx_bridge")
    p.add_argument("--task", default="widowx_put_eggplant_in_basket")
    p.add_argument("--n-episodes", type=int, default=8)
    p.add_argument("--output-dir", default="./tome_spatialvla_results")
    p.add_argument("--tome", action="store_true", default=False)
    p.add_argument("--tome-r", type=int, default=8)
    p.add_argument("--tome-layers", type=int, default=6)
    p.add_argument("--tome-protect", default="none", choices=["none", "center"])
    p.add_argument("--tome-protect-ratio", type=float, default=0.25)
    p.add_argument("--temporal-stride", type=int, default=1,
                   help="reuse SigLIP features for stride-1 steps (1 = off)")
    p.add_argument("--depth-prune", type=int, default=0,
                   help="bypass N most-redundant Gemma2 decoder layers (0 = off)")
    p.add_argument("--depth-prune-min-layer", type=int, default=2,
                   help="protect the first M layers from pruning")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    cfg = TASK_CONFIGS[args.task]
    cam = cfg["obs_camera_name"]

    policy = SpatialVLAInference(
        saved_model_path=args.model_path,
        unnorm_key=args.unnorm_key,
        policy_setup=args.policy_setup,
    )
    print(f"[OK] official SpatialVLAInference loaded (no wrapper, no DINO)", flush=True)

    if args.tome:
        base_model = getattr(policy, "vla", None) or getattr(policy, "model", None)
        vision_tower = getattr(base_model, "vision_tower", None)
        if vision_tower is None:
            raise AttributeError("could not locate SpatialVLA vision_tower for ToMe")
        protect = (center_protect_provider(args.tome_protect_ratio)
                   if args.tome_protect == "center" else None)
        apply_tome_to_siglip(vision_tower, r=args.tome_r,
                             num_merge_layers=args.tome_layers, protect_provider=protect)

    if args.temporal_stride > 1:
        from temporal_cache import apply_temporal_cache, reset_temporal_cache
        base_model = getattr(policy, "vla", None) or getattr(policy, "model", None)
        apply_temporal_cache(base_model, stride=args.temporal_stride)

    pruner = None
    if args.depth_prune > 0:
        from depth_prune_gemma2 import DepthPruner
        base_model = getattr(policy, "vla", None) or getattr(policy, "model", None)
        pruner = DepthPruner(base_model.language_model)
        print(f"[DepthPrune] Gemma2 has {pruner.n} decoder layers; "
              f"will bypass {args.depth_prune} most-redundant (calibrate on step 0).",
              flush=True)

    base_ids = list(range(*cfg["obj_episode_range"]))
    ep_ids = [base_ids[i % len(base_ids)] for i in range(args.n_episodes)]
    results = []
    calibrated = False
    if pruner is not None:
        pruner.install_calibration_hooks()

    for ep_count, ep_id in enumerate(ep_ids):
        print(f"\n── ep {ep_count:02d} (env_id={ep_id}) ──", flush=True)
        env, obs = build_env(cfg, ep_id)
        instruction = env.get_language_instruction()
        image = get_image(env, obs, cam)
        policy.reset(instruction)
        if args.temporal_stride > 1:
            from temporal_cache import reset_temporal_cache
            reset_temporal_cache(getattr(policy, "vla", None) or getattr(policy, "model", None))
        print(f"   instruction: {instruction}", flush=True)

        done = truncated = False
        step = 0
        model_time = 0.0
        model_calls = 0
        grasped = False
        final_info = {}
        while not (done or truncated) and step < cfg["max_episode_steps"]:
            t0 = time.time()
            raw_action, action = policy.step(image, instruction)
            model_time += time.time() - t0
            model_calls += 1
            if pruner is not None and not calibrated:
                pruner.finalize_calibration()
                bypassed = pruner.apply(args.depth_prune, args.depth_prune_min_layer)
                print(f"[DepthPrune] calibrated; bypassing layers {bypassed} "
                      f"(ranking top: {pruner.ranking[:6]})", flush=True)
                calibrated = True
            env_action = np.concatenate([
                action["world_vector"], action["rot_axangle"],
                np.atleast_1d(action["gripper"]),
            ])
            obs, _, done, truncated, info = env.step(env_action)
            final_info = info
            if not grasped and isinstance(info, dict) and info.get("is_src_obj_grasped", False):
                grasped = True
            image = get_image(env, obs, cam)
            new_instr = env.get_language_instruction()
            if new_instr != instruction:
                instruction = new_instr
                policy.reset(instruction)
            step += 1

        success = bool(final_info.get("success", done))
        grasped = grasped or bool(final_info.get("ever_grasped_src", False))
        model_ms = (model_time / model_calls * 1000.0) if model_calls else 0.0
        g_mark = "G+" if grasped else "G-"
        print(f"   → {g_mark} {'SUCCESS' if success else 'FAIL'}  "
              f"({step} steps, {model_ms:.0f} ms/infer)", flush=True)
        results.append({"ep": ep_count, "ep_id": ep_id, "success": success,
                        "grasped": grasped, "steps": step, "model_ms_per_infer": model_ms})
        env.close()

    n_ok = sum(r["success"] for r in results)
    n_grasp = sum(r["grasped"] for r in results)
    sr = n_ok / len(results)
    gr = n_grasp / len(results)
    avg_ms = float(np.mean([r["model_ms_per_infer"] for r in results]))
    avg_steps = float(np.mean([r["steps"] for r in results]))
    parts = []
    if args.tome:
        parts.append(f"ToMe r={args.tome_r}x{args.tome_layers} protect={args.tome_protect}")
    if args.temporal_stride > 1:
        parts.append(f"temporal-stride={args.temporal_stride}")
    if args.depth_prune > 0 and pruner is not None:
        parts.append(f"depth-prune={args.depth_prune} (layers {pruner.pruned})")
    tag = " + ".join(parts) if parts else "baseline (no ToMe)"
    print(f"\n{'='*50}")
    print(f"  SpatialVLA (official) | {tag}")
    print(f"  task:    {args.task}")
    print(f"  파지율:  {n_grasp}/{len(results)} = {gr:.1%}")
    print(f"  성공률:  {n_ok}/{len(results)} = {sr:.1%}")
    print(f"  평균스텝: {avg_steps:.0f}")
    print(f"  ms/infer: {avg_ms:.0f}  (latency; lower=faster)")
    print(f"{'='*50}")
    for r in results:
        gm = "G+" if r["grasped"] else "G-"
        sm = "OK" if r["success"] else "--"
        print(f"  {sm}{gm} ep{r['ep']:02d} (id={r['ep_id']}): {r['steps']} steps")
    print("", flush=True)

    summary = {
        "model": "SpatialVLA-official", "task": args.task,
        "success_rate": sr, "grasp_rate": gr,
        "avg_model_ms_per_infer": avg_ms, "avg_steps": avg_steps,
        "tome": {"enabled": args.tome, "r": args.tome_r, "layers": args.tome_layers,
                 "protect": args.tome_protect},
        "temporal_stride": args.temporal_stride,
        "depth_prune": ({"count": args.depth_prune, "min_layer": args.depth_prune_min_layer,
                         "pruned": pruner.pruned, "ranking": pruner.ranking}
                        if pruner is not None else {"count": 0}),
        "episodes": results,
    }
    with open(os.path.join(args.output_dir, f"results_{args.task}.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"saved: {args.output_dir}/results_{args.task}.json", flush=True)


if __name__ == "__main__":
    main()

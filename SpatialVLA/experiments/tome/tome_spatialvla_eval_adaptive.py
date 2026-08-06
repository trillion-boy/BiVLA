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
from spatialvla_eval import (TASK_CONFIGS, build_env, get_image,  # noqa: E402
                             episode_grasped, step_grasped)

# official SpatialVLA policy (no wrapper)
from simpler_env.policies.spatialvla.spatialvla_model import SpatialVLAInference  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
from tome_siglip import apply_tome_to_siglip, center_protect_provider  # noqa: E402
from adaptive_chunk_exec import AdaptiveChunkExecutor  # noqa: E402
from foveation import foveate_image_logpolar  # noqa: E402


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
    p.add_argument("--depth-prune-mode", default="redundant",
                   choices=["redundant", "least", "random"],
                   help="which N layers to bypass -- 'redundant' is the real method;"
                        " 'least'/'random' are controls to sanity-check the ranking")
    p.add_argument("--depth-prune-seed", type=int, default=0)
    p.add_argument("--depth-prune-min-layer", type=int, default=2,
                   help="protect the first M layers from pruning")
    p.add_argument("--spec-decode", action="store_true", default=False,
                   help="self-speculative decoding (lossless -- see "
                        "gemma2_self_spec_decode.py); mutually exclusive with --depth-prune")
    p.add_argument("--spec-decode-gamma", type=int, default=4)
    p.add_argument("--spec-decode-layer-count", type=int, default=4,
                   help="how many top-redundant layers the draft pass bypasses")
    p.add_argument("--spec-decode-cache-budget", type=int, default=64,
                   help="HybridCache is sized prompt_len+this (not +max_new_tokens=256) "
                        "to cut clone cost; raise if a WARNING about hitting the budget appears")
    p.add_argument("--exec-chunk", type=int, default=0,
                   help="execute k of the model's predicted action chunk per generate "
                        "(0 = off = re-generate every step; k>chunk_size clamps to chunk_size). "
                        "Decode is called every k env steps -> latency amortized ~k x.")
    p.add_argument("--adaptive-chunk", action="store_true", default=False,
                   help="enable adaptive chunk execution: k=4 during transport, k=1 during grasp/place. "
                        "Detects phase via gripper state. Incompatible with --exec-chunk.")
    p.add_argument("--adaptive-k-sparse", type=int, default=4,
                   help="chunk size during transport/approach phase")
    p.add_argument("--adaptive-k-dense", type=int, default=1,
                   help="chunk size during grasp/place phase")
    p.add_argument("--adaptive-close-delay", type=int, default=3,
                   help="steps after gripper closes before switching to dense replanning")
    p.add_argument("--adaptive-open-delay", type=int, default=5,
                   help="steps after gripper opens before switching back to sparse replanning")
    p.add_argument("--foveate", action="store_true", default=False,
                   help="log-polar foveation of the observation before the policy sees it "
                        "(mentor's RetinaBased transform ported 1:1); composes with any "
                        "other flag -- the env always steps on the raw scene")
    p.add_argument("--foveate-keep-percent", type=float, default=20.0,
                   help="percent of visual sample density retained (RetinaBased default: 20)")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    cfg = TASK_CONFIGS[args.task]
    cam = cfg["obs_camera_name"]

    fov_keep_ratio = args.foveate_keep_percent / 100.0

    def observe(env, obs):
        img = get_image(env, obs, cam)
        return foveate_image_logpolar(img, fov_keep_ratio) if args.foveate else img

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

    spec_pruner = None
    if args.spec_decode:
        from depth_prune_gemma2 import DepthPruner
        base_model = getattr(policy, "vla", None) or getattr(policy, "model", None)
        spec_pruner = DepthPruner(base_model.language_model)
        print(f"[SpecDecode] Gemma2 has {spec_pruner.n} decoder layers; "
              f"draft will bypass top-{args.spec_decode_layer_count} redundant "
              f"(gamma={args.spec_decode_gamma}, calibrate on step 0, lossless).",
              flush=True)

    chunk_state = None
    adaptive_executor = None
    if args.adaptive_chunk and args.exec_chunk > 0:
        raise ValueError("--adaptive-chunk and --exec-chunk are mutually exclusive")
    if args.adaptive_chunk:
        adaptive_executor = AdaptiveChunkExecutor(
            policy,
            k_sparse=args.adaptive_k_sparse,
            k_dense=args.adaptive_k_dense,
            close_steps_before_dense=args.adaptive_close_delay,
            open_steps_after_release=args.adaptive_open_delay,
        )
        print(f"[AdaptiveChunk] initialized: k_sparse={args.adaptive_k_sparse}, "
              f"k_dense={args.adaptive_k_dense}, "
              f"close_delay={args.adaptive_close_delay}, "
              f"open_delay={args.adaptive_open_delay}", flush=True)
    elif args.exec_chunk > 0:
        from chunk_exec import apply_chunk_execution, reset_chunk_execution
        chunk_state = apply_chunk_execution(policy, k=args.exec_chunk)

    base_ids = list(range(*cfg["obj_episode_range"]))
    ep_ids = [base_ids[i % len(base_ids)] for i in range(args.n_episodes)]
    results = []
    calibrated = False
    if pruner is not None:
        pruner.install_calibration_hooks()
    if spec_pruner is not None:
        spec_pruner.install_calibration_hooks()

    for ep_count, ep_id in enumerate(ep_ids):
        print(f"\n── ep {ep_count:02d} (env_id={ep_id}) ──", flush=True)
        env, obs = build_env(cfg, ep_id, task_name=args.task)
        instruction = env.get_language_instruction()
        image = observe(env, obs)
        policy.reset(instruction)
        if args.temporal_stride > 1:
            from temporal_cache import reset_temporal_cache
            reset_temporal_cache(getattr(policy, "vla", None) or getattr(policy, "model", None))
        if adaptive_executor is not None:
            adaptive_executor.reset()
        elif chunk_state is not None:
            from chunk_exec import reset_chunk_execution
            reset_chunk_execution(policy)
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
            if ep_count == 0 and step == 0:
                # One-time probe: does SpatialVLA already predict an action
                # CHUNK per generate call (raw_action holding >1 timestep)?
                # If yes, executing the whole chunk instead of re-generating
                # every env step is the remaining decode-latency lever
                # (decode is 75% of a step and cannot be made cheaper on this
                # backbone -- only called less often).
                for name, d in (("raw_action", raw_action), ("action", action)):
                    if isinstance(d, dict):
                        for k, v in d.items():
                            shape = getattr(v, "shape", None)
                            print(f"[ChunkProbe] {name}['{k}']: shape={shape} "
                                  f"type={type(v).__name__}", flush=True)
                    else:
                        print(f"[ChunkProbe] {name}: type={type(d).__name__} "
                              f"shape={getattr(d, 'shape', None)}", flush=True)
            if pruner is not None and not calibrated:
                pruner.finalize_calibration()
                bypassed = pruner.apply(args.depth_prune, args.depth_prune_min_layer,
                                        mode=args.depth_prune_mode, seed=args.depth_prune_seed)
                red_str = ", ".join(f"L{i}={pruner._redundancy[i]:.4f}" for i in range(pruner.n))
                print(f"[DepthPrune] mode={args.depth_prune_mode} calibrated; "
                      f"bypassing layers {bypassed} (ranking top: {pruner.ranking[:6]})", flush=True)
                print(f"[DepthPrune] per-layer redundancy (cos in/out): {red_str}", flush=True)
                calibrated = True
            if spec_pruner is not None and not calibrated:
                from spec_decode_gemma2 import apply_gemma2_self_spec_decode
                spec_pruner.finalize_calibration()
                spec_pruner.restore()  # spec-decode toggles bypass/restore itself per round
                draft_layers = spec_pruner.ranking[: args.spec_decode_layer_count]
                base_model = getattr(policy, "vla", None) or getattr(policy, "model", None)
                gen_cfg = getattr(base_model, "generation_config", None)
                eos_id = getattr(gen_cfg, "eos_token_id", None) if gen_cfg is not None else None
                apply_gemma2_self_spec_decode(
                    base_model, spec_pruner, args.spec_decode_gamma, draft_layers, eos_token_id=eos_id,
                    cache_len_budget=args.spec_decode_cache_budget,
                )
                print(f"[SpecDecode] calibrated; draft bypasses layers {draft_layers} "
                      f"(ranking top: {spec_pruner.ranking[:6]}, eos_token_id={eos_id}, "
                      f"cache_budget={args.spec_decode_cache_budget})", flush=True)
                calibrated = True
            # Update adaptive executor with action for phase detection
            if adaptive_executor is not None:
                adaptive_executor.step(action)

            env_action = np.concatenate([
                action["world_vector"], action["rot_axangle"],
                np.atleast_1d(action["gripper"]),
            ])
            obs, _, done, truncated, info = env.step(env_action)
            final_info = info
            grasped = grasped or step_grasped(info)
            image = observe(env, obs)
            new_instr = env.get_language_instruction()
            if new_instr != instruction:
                instruction = new_instr
                policy.reset(instruction)
                if adaptive_executor is not None:
                    adaptive_executor.reset()
                elif chunk_state is not None:
                    from chunk_exec import reset_chunk_execution
                    reset_chunk_execution(policy)  # task changed -> stale plan, replan now
            step += 1

        success = bool(final_info.get("success", done))
        grasped = grasped or episode_grasped(final_info)
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
    if args.foveate:
        parts.append(f"foveated keep={args.foveate_keep_percent:.0f}%")
    if args.tome:
        parts.append(f"ToMe r={args.tome_r}x{args.tome_layers} protect={args.tome_protect}")
    if args.temporal_stride > 1:
        parts.append(f"temporal-stride={args.temporal_stride}")
    if args.depth_prune > 0 and pruner is not None:
        parts.append(f"depth-prune={args.depth_prune} mode={args.depth_prune_mode} "
                     f"(layers {pruner.pruned})")
    spec_stats = None
    if args.spec_decode and spec_pruner is not None:
        base_model = getattr(policy, "vla", None) or getattr(policy, "model", None)
        spec_stats = getattr(base_model, "_specdec_stats", None)
        acc_rate = (spec_stats["accepted"] / spec_stats["proposed"]
                   if spec_stats and spec_stats.get("proposed") else None)
        tag_str = f"spec-decode gamma={args.spec_decode_gamma} layers={spec_pruner.ranking[:args.spec_decode_layer_count]}"
        if acc_rate is not None:
            tag_str += f" (acceptance={acc_rate:.1%})"
        parts.append(tag_str)
    if adaptive_executor is not None:
        parts.append(f"adaptive-chunk k_sparse={args.adaptive_k_sparse} k_dense={args.adaptive_k_dense}")
    elif chunk_state is not None:
        ratio = (chunk_state["gen_calls"] / chunk_state["steps"]
                 if chunk_state["steps"] else 1.0)
        parts.append(f"exec-chunk k={chunk_state['k']}/{chunk_state['chunk_size']} "
                     f"(generate on {ratio:.0%} of steps)")
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
        "foveate": {"enabled": args.foveate,
                    "keep_percent": args.foveate_keep_percent},
        "tome": {"enabled": args.tome, "r": args.tome_r, "layers": args.tome_layers,
                 "protect": args.tome_protect},
        "temporal_stride": args.temporal_stride,
        "depth_prune": ({"count": args.depth_prune, "min_layer": args.depth_prune_min_layer,
                         "mode": args.depth_prune_mode, "seed": args.depth_prune_seed,
                         "pruned": pruner.pruned, "ranking": pruner.ranking,
                         "redundancy": [round(x, 4) for x in pruner._redundancy]}
                        if pruner is not None else {"count": 0}),
        "spec_decode": ({"enabled": True, "gamma": args.spec_decode_gamma,
                         "layer_count": args.spec_decode_layer_count,
                         "draft_layers": spec_pruner.ranking[:args.spec_decode_layer_count],
                         "ranking": spec_pruner.ranking, "stats": spec_stats}
                        if args.spec_decode and spec_pruner is not None else {"enabled": False}),
        "exec_chunk": ({"enabled": True, "k": chunk_state["k"],
                        "chunk_size": chunk_state["chunk_size"],
                        "gen_calls": chunk_state["gen_calls"], "steps": chunk_state["steps"]}
                       if chunk_state is not None else {"enabled": False}),
        "adaptive_chunk": ({"enabled": True, "k_sparse": args.adaptive_k_sparse,
                            "k_dense": args.adaptive_k_dense,
                            "close_delay": args.adaptive_close_delay,
                            "open_delay": args.adaptive_open_delay}
                           if adaptive_executor is not None else {"enabled": False}),
        "episodes": results,
    }
    with open(os.path.join(args.output_dir, f"results_{args.task}.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"saved: {args.output_dir}/results_{args.task}.json", flush=True)


if __name__ == "__main__":
    main()

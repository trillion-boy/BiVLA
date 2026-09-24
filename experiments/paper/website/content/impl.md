Section III of the paper defines the five tricks and Section IV-A the protocol. This section adds the exact settings and the values every run recorded. All numbers come from the 700 run files (22 backbone and environment pairs x 14 configurations, one file per task or suite folder).

### 1.1 Trick settings

**Foveation** (Eq. 2). Keep ratio 0.2 or 0.5 of the image area, fovea at the image centre, sharp-disc radius r = sqrt(keep ratio x H x W / pi) (about 140 px on a 640 x 480 frame at keep 0.2). Outside the disc the pixel is a blend of the input and two Gaussian blurs with sigma 3 and 9 px, with a ramp that rises from 0 at the disc edge to 1 at the farthest image corner: for tau in [0, 0.5] the pixel is (1 - 2 tau) I + 2 tau G3, for tau in [0.5, 1] it is (2 - 2 tau) G3 + (2 tau - 1) G9. Pixels inside the disc are copied back unchanged. Foveation is applied at every step to the image given to the policy, at the camera's native resolution; the image size and the visual-token count are unchanged, and the blur is a CPU cost on every frame.

![Foveation examples](figs/foveation_examples.png)
*Top: raw WidowX observations. Bottom: the same frames at keep ratio 0.2.*

**Action repeat.** The policy is called once and each returned action is executed k times (`[a, b]` becomes `[a, a, b, b]` on a chunk); the policy is not queried during repeated steps and the next call sees the latest observation. Every simulator step counts as an environment step. UniVLA emits a chunk of 5 actions on WidowX and 10 on LIBERO, so its open-loop horizon at k = 2 is 10 and 20 steps and at k = 4 is 20 and 40 steps (recorded steps per call: 9.9 / 19.9 on WidowX, 19.8 / 38.8 on LIBERO). All other backbones execute one action per call, so k = 2 is a 2-step horizon.

**Depth pruning.** Block Influence (Eq. 4) is measured with a forward hook on every decoder layer on the prefill of the first policy call of the run (all prompt positions); that call runs unpruned and the ranking is then frozen for the run. On LIBERO each suite is a separate run. CronusVLA runs a separate calibration pass over the environment's task list (seed 10000) on its 12-layer DiT action decoder. Layers in the first quarter of the stack and the final layer are never removed, and no two removed layers are adjacent; all 138 recorded selections (54 distinct pair-and-budget selections) satisfy these rules. A removed block is replaced by a pass-through that keeps the KV cache contiguous, so the block's attention and MLP are not executed on any token. The layers actually removed:

| Backbone and environment | Layers | 1 layer | 2 layers | 4 layers |
|---|---|---|---|---|
| CogACT WidowX | 32 | 17 | 17, 23 | 17, 20, 23, 25 |
| CogACT Fractal | 32 | 23 | 17, 23 | 17, 19, 23, 25 |
| OpenVLA WidowX and Fractal | 32 | 23 | 23, 25 | 17, 23, 25, 27 |
| OpenVLA LIBERO Long | 32 | 23 | 23, 25 | 19, 21, 23, 25 |
| OpenVLA LIBERO Goal | 32 | 17 | 17, 20 | 8, 17, 20, 23 |
| OpenVLA LIBERO Object | 32 | 17 | 17, 19 | 17, 19, 21, 23 |
| OpenVLA LIBERO Spatial | 32 | 8 | 8, 20 | 8, 20, 23, 25 |
| SpatialVLA WidowX and Fractal | 26 | 8 | 8, 10 | 8, 10, 13, 23 |
| CronusVLA WidowX (DiT) | 12 | 10 | 8, 10 | 4, 6, 8, 10 |
| CronusVLA Fractal (DiT) | 12 | 10 | 8, 10 | 3, 6, 8, 10 |
| UniVLA WidowX | 32 | 26 | 26, 30 | 21, 24, 26, 30 |
| UniVLA LIBERO Long | 32 | 19 | 19, 21 | 19, 21, 26, 30 |
| UniVLA LIBERO Goal and Object | 32 | 19 | 19, 21 | 19, 21, 26, 29 |
| UniVLA LIBERO Spatial | 32 | 19 | 19, 22 | 19, 22, 26, 29 |
| MiniVLA WidowX | 24 | 13 | 11, 13 | 7, 9, 11, 13 |
| SmolVLA LIBERO (all suites) | 32 | 30 | 28, 30 | 24, 26, 28, 30 |

SmolVLA's layers are fixed late indices without a Block Influence measurement.

**Guarded reuse** (Eq. 5). The previous action is repeated for one step only when all gates pass; otherwise the policy is queried in full. Thresholds of the three presets, identical on every backbone that recorded them:

| Gate | Statistic | Strict | Moderate | Aggressive |
|---|---|---|---|---|
| Global image change | mean absolute difference between signatures of consecutive observations, at most | 0.01 | 0.015 | 0.02 |
| Local patch change | maximum over local patches of the same difference, at most | 0.03 | 0.04 | 0.05 |
| Action agreement | cosine similarity of the two most recent inferred 6-D pose actions, at least | 0.995 | 0.99 | 0.98 |
| Translation floor | translation norm of the candidate action, at least | 0.01 | 0.01 | 0.01 |
| Gripper | commanded gripper state unchanged | yes | yes | yes |
| Reuse cap | consecutive reuses, fewer than | 1 | 1 | 2 |

**Temporal fusion** (Eq. 8). Patches with motion above 0.01, the 15 percent highest-entropy patches, the 20 percent highest text-to-vision-attention patches (task-aware only) and their 1-patch neighbourhood are recomputed; the rest is reused from the previous call up to a cap. Motion-entropy and task-aware: cap 0.5 of the patches, a full keyframe every 3rd call. Conservative-adaptive: cap 0.25, a keyframe every 2nd call and a forced keyframe when frame motion exceeds 0.03. Fusion acts on the projected visual tokens before the language decoder (CogACT, SpatialVLA, OpenVLA, MiniVLA) or on the discrete VQ codes (UniVLA); it does not change the number of policy calls. Collecting the attention for the task-aware setting forces an SDPA decoder into eager attention.

### 1.2 How often the gated tricks acted

A gated trick can only change a result where it fires, so the fire counts are part of the result.

| Backbone and environment | Reused steps, strict / moderate / aggressive (%) | Keyframe share, motion-entropy / task-aware / conservative | Median reused tokens |
|---|---|---|---|
| CogACT WidowX | 0.3 / 0.9 / 2.7 | 0.34 / same run / 0.93 | 108 / 108 / 64 |
| CogACT Fractal | 0.01 / 0.1 / 0.6 | 0.34 / same run / 0.91 | 85 to 128 / same / 64 |
| OpenVLA WidowX | 3.7 / 4.8 / 6.3 | 0.34 / 0.34 / 0.93 | 78 to 128 / 46 to 91 / 64 |
| OpenVLA Fractal | 2.3 / 2.5 / 3.9 | 0.34 / 0.34 / 0.82 | 102 to 117 / 47 to 62 / 64 |
| OpenVLA LIBERO (four suites) | 2.0 to 4.8 / 3.5 to 5.6 / 9.0 to 11.2 | 0.33 to 0.34 / 0.33 to 0.34 / 0.85 to 0.93 | 124 to 128 / 61 to 77 / 64 |
| SpatialVLA WidowX | 0.02 / 0.07 / 0.4 | 0.34 / 0.34 / 0.98 | 57 to 108 / 26 to 54 / 0 |
| SpatialVLA Fractal | 0.5 / 0.8 / 2.4 | 0.34 / 0.34 / 0.99 | 69 to 118 / 19 to 48 / 0 |
| CronusVLA WidowX | 0.05 / 0.3 / 1.4 | one run for all three settings, 106 fused patches per call | |
| CronusVLA Fractal | 0.7 / 1.7 / 4.2 | one run for all three settings, 112 fused patches per call | |
| UniVLA WidowX | 0.0 / 0.02 / 0.03 | 0.37 / not recorded / 1.00 | 314 to 442 / not recorded / 0 |
| UniVLA LIBERO (four suites) | 0.0 to 0.06 in every cell | 0.35 to 0.37 / 0.35 to 0.36 / 1.00 | 300 to 312 / 141 to 168 / 0 |
| MiniVLA WidowX | 3.1 / 4.2 / 4.0 | 0.34 / 0.34 / 0.90 | 107 / 68 / 64 |
| SmolVLA LIBERO (four suites) | 0.02 to 0.13 / 0.2 to 0.5 / 1.4 to 2.6 | not recorded | 19 to 23 / 3.5 to 5 / 16 |

Consequences for reading the tables: the gates open on about a tenth of the steps at most (OpenVLA LIBERO, aggressive) and on under 1 percent on CogACT Fractal, SpatialVLA WidowX and every UniVLA cell; on UniVLA Object, Spatial and WidowX strict no gate ever fired and those runs are identical to the original. Conservative-adaptive fusion reused no patch on SpatialVLA and UniVLA (every call a keyframe), so those cells are the original policy under the fusion name. CogACT exposes no text-to-vision attention where fusion runs, so its task-aware setting is its motion-entropy run. CronusVLA's three fusion settings are one run.

### 1.3 Harness, checkpoints, software and hardware

**Episodes and seeds.** WidowX: 4 tasks x 50 episodes, step cap 60 (120 for eggplant in basket). Fractal: 5 tasks x 50 episodes, cap 80. LIBERO Long, Goal, Object, Spatial: 10 tasks x 10 episodes, caps 520, 300, 280, 220. Every configuration replays the same task instances with seed 42 plus the episode index. A failed episode runs to its cap (all 21,568 failed episodes end exactly at the cap), so average steps follows success. Per pair: 200 episodes per WidowX configuration, 250 per Fractal configuration, 100 per LIBERO suite and configuration; 47,600 episodes in all.

**Checkpoints and per-backbone settings.**

| Backbone | Checkpoint | Decoder pruned | Actions per call | Attention and versions |
|---|---|---|---|---|
| CogACT | CogACT-Base (Llama-2-7B base); DDIM 10 steps, cfg 1.5 | 32 Llama layers | 1 | Llama decoder resolves to SDPA under transformers 4.47.0 |
| OpenVLA | openvla-7b (SimplerEnv); openvla-7b-finetuned-libero-spatial / -object / -goal / -10 (LIBERO, one per suite); 224 px, no centre crop | 32 Llama-2 layers | 1 | SDPA requested on SimplerEnv, eager on LIBERO |
| SpatialVLA | spatialvla-4b-224-sft-bridge and -sft-fractal (PaliGemma2-3B); native chunk 4 with action ensembling, queried every step | 26 Gemma2 layers | 1 | text decoder eager (Gemma2 soft-capping), transformers 4.47.0 |
| CronusVLA | released checkpoint (training step 42,500) | 12-layer DiT action decoder | 1 | Qwen2.5 decoder resolves to SDPA under transformers 4.47.0 |
| UniVLA | UNIVLA_SIMPLER_BRIDGE_VIDEO_BS128_20K (WidowX), UNIVLA_LIBERO_VIDEO_BS192_8K (LIBERO), Emu3 vision tokenizer | 32 Emu3 layers | chunk of 5 (WidowX) or 10 (LIBERO) | SDPA |
| MiniVLA | minivla-vq-bridge-prismatic (prism-qwen25-extra-dinosiglip-224px, 0.5B); 224 px, no centre crop | 24 Qwen2.5 layers | 1 | resolves to SDPA under transformers 4.47.0 |
| SmolVLA | LIBERO fine-tuned SmolVLA on SmolVLM2-500M-Instruct | 32 SmolLM2 layers (fixed indices) | 1 | explicit PyTorch SDPA; lerobot 0.4.4, transformers 4.51.3 |

**GPU cards.** The runs were scheduled on a shared GPU cluster. The run files record the card per episode for the runs made after 6 September: every such run used an RTX 5090 (the SmolVLA guarded reuse and temporal fusion cells, the CronusVLA WidowX moderate and aggressive reuse cells and the UniVLA WidowX task-aware fusion cell), except that in the SmolVLA depth-pruning cells at two layers on Long and Goal and at four layers on Long, Goal and Object, 86 of the 500 episodes list an RTX PRO 6000, RTX 6000 Ada, L40S or RTX A6000. Latency is compared only within one backbone and environment.

**Two implementations of SmolVLA.** SmolVLA's guarded reuse, temporal fusion and the depth cells named above were completed with a re-implemented evaluator that is 2 to 7 percent faster per call than the original implementation even where a trick does nothing (strict reuse -3.3 to -6.2 percent, conservative fusion -2.3 to -5.5). Their latency values mix that implementation difference with the trick and are read through the paired success test alone.

**Latency.** Wall-clock per environment step: mean episode time divided by mean episode length. The episode clock runs from reset to termination or cap and includes simulator stepping and rendering, the policy calls, the foveation blur, the reuse gate and the fusion selector; it excludes model loading, environment construction and CronusVLA's separate calibration pass. Every harness except the original CronusVLA runs also records the per-call model time, used in Section 3.

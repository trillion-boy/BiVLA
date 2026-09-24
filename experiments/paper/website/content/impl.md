Section III of the paper defines the five tricks and Section IV-A the protocol. This section adds the concrete settings, the values every run recorded, and the harness conventions that the tables depend on. Sources are the code of the trick modules and the per-episode records of the 700 runs (one `summary.json` and one `episodes.jsonl` per backbone, environment and configuration).

### 1.1 Foveation

![Foveation examples](figs/foveation_examples.png)
*Top: raw WidowX observations (eggplant in basket, carrot on plate, spoon on towel, stack cube). Bottom: the same frames after foveation at keep ratio 0.2. The central disc is untouched, the periphery is a blend of two Gaussian blurs.*

The foveated observation of Eq. (2) is built as follows, at the camera's native resolution, before the policy's own resize.

| Quantity | Value |
|---|---|
| Keep ratio | 0.2 or 0.5 of the image area (`fixed_foveation_keep20`, `fixed_foveation_keep50`) |
| Fovea centre | image centre |
| Sharp-disc radius | r = sqrt(keep ratio x H x W / pi), about 140 px on a 640 x 480 frame at keep 0.2 |
| Ramp | tau rises linearly from 0 at the disc edge to 1 at the farthest image corner |
| Blurs | two Gaussian blurs with sigma 3 px and sigma 9 px |
| Weights | for tau in [0, 0.5] the pixel is (1 - 2 tau) I + 2 tau G3; for tau in [0.5, 1] it is (2 - 2 tau) G3 + (2 tau - 1) G9 |
| Fovea | pixels inside the disc are copied back from the input, so the disc is bit-identical |

Foveation is applied at every step to the image given to the policy; the simulator keeps stepping on the raw frame. The image size and the visual-token count are unchanged, so the model does the same work, and the blur is an extra CPU cost on every frame (about 11 ms per 256 x 256 frame and 45 ms per 640 x 480 frame on a workstation CPU). Its effect on per-step latency is quantified in Section 3. On LIBERO the agent-view image is foveated; whether the wrist view was also foveated is not recorded in the run files.

### 1.2 Action repeat

The policy is called once and each action it returned is executed k times before the next call (`[a, b]` becomes `[a, a, b, b]` on a chunk). The simulator still returns an observation after every step, the policy is not queried during repeated steps, and the next call sees the latest observation. Every simulator step counts as one environment step and every call as one policy call, so for a single-action policy the number of calls is ceil(steps / k).

UniVLA emits a chunk per call (5 actions on WidowX, 10 on LIBERO), so the same nominal k is a longer open-loop horizon there. The records confirm it:

| Backbone and environment | k | Steps per episode | Calls per episode | Steps per call |
|---|---|---|---|---|
| UniVLA WidowX | 1 (original) | 30.21 | 6.38 | 4.7 |
| UniVLA WidowX | 2 | 71.93 | 7.25 | 9.9 |
| UniVLA WidowX | 4 | 74.52 | 3.74 | 19.9 |
| UniVLA LIBERO (four suites) | 1 | 164.27 | 16.84 | 9.8 |
| UniVLA LIBERO | 2 | 296.07 | 14.93 | 19.8 |
| UniVLA LIBERO | 4 | 329.33 | 8.48 | 38.8 |
| CogACT WidowX | 2 | 70.61 | 35.33 | 2.0 |
| OpenVLA WidowX | 2 | 60.49 | 30.33 | 2.0 |
| OpenVLA LIBERO | 2 | 224.66 | 112.50 | 2.0 |
| SmolVLA LIBERO | 2 | 231.22 | 115.77 | 2.0 |

The repeat loop stops at the episode's termination or step cap. The `reuses` counter recorded under action repeat is harness specific (0 on CogACT and SpatialVLA, steps minus calls on OpenVLA, MiniVLA, SmolVLA and CronusVLA, (k - 1) x chunk x calls on UniVLA), so only `policy_calls` and `steps_executed` are compared across backbones.

### 1.3 Depth pruning

**Block Influence.** A forward hook on every decoder layer takes the layer's input and output hidden states on the prefill of the first policy call (all prompt positions, image and text tokens), computes the cosine similarity along the hidden dimension per token, averages over positions, and stores 1 minus the mean. The single-token decode calls are not included. For CogACT, MiniVLA, OpenVLA, SpatialVLA and UniVLA this measurement runs once per run on the first test observation, whose own step runs unpruned; the ranking is then frozen for the run. On LIBERO each suite is a separate run with its own checkpoint, so the ranking is per suite. CronusVLA runs a separate calibration pass over the environment's task list (seed 10000) on its 12-layer DiT action decoder.

**Protection rules.** Layers in the first quarter of the stack are never removed, the final layer is never removed, and no two removed layers are adjacent. All 120 recorded selections satisfy these rules. The rule was binding: on every 32-layer stack the lowest-influence layers are early ones (for OpenVLA on WidowX the six lowest are 3, 2, 5, 4, 23, 6), so the removed layers are the lowest-influence layers outside the protected window.

**Mechanism.** A removed block is replaced in the module list by a pass-through that returns the hidden states unchanged and, when a KV cache is in use, writes a zero placeholder at its layer index so the cache stays contiguous. The block's attention and MLP are not executed on prefill or on any decode token. Weights are untouched.

**Layers actually removed** (from `depth_calibration.selected_layers`):

| Backbone and environment | Layers | 1 layer | 2 layers | 4 layers |
|---|---|---|---|---|
| CogACT WidowX | 32 | 17 | 17, 23 | 17, 20, 23, 25 |
| CogACT Fractal | 32 | 23 | 17, 23 | 17, 19, 23, 25 |
| OpenVLA WidowX | 32 | 23 | 23, 25 | 17, 23, 25, 27 |
| OpenVLA Fractal | 32 | 23 | 23, 25 | 17, 23, 25, 27 |
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

SmolVLA's layers are fixed indices with no Block Influence measurement (`calibrated: false`), so its depth-pruning rows test late-layer removal rather than the calibrated procedure of Section III-D.

### 1.4 Guarded reuse

The reuse decision has six conjuncts. The thresholds of the three presets are identical on every backbone that recorded them:

| Gate | Statistic | Strict | Moderate | Aggressive |
|---|---|---|---|---|
| Global image change | mean absolute difference between cheap signatures of consecutive observations, at most | 0.01 | 0.015 | 0.02 |
| Local patch change | maximum over local patches of the same difference, at most | 0.03 | 0.04 | 0.05 |
| Action agreement | cosine similarity of the two most recent inferred 6-D pose actions, at least | 0.995 | 0.99 | 0.98 |
| Translation floor | translation norm of the candidate action, at least | 0.01 | 0.01 | 0.01 |
| Gripper | commanded gripper state unchanged | yes | yes | yes |
| Reuse cap | consecutive reuses so far, fewer than | 1 | 1 | 2 |

When any gate fails the policy is queried in full and the fresh action is executed. When all pass, the previous executed action is repeated for one step and the observation is examined again at the next step; with a cap of 1 a reused step is always followed by a full call. Every reused step is counted in `reuses` and every full call in `policy_calls`, so calls plus reuses equal the steps on single-action policies. The strict values are also the harness defaults.

**Gate activity.** The share of steps on which the gate opened decides how much a reuse cell can differ from the original. It is small almost everywhere:

<details markdown="1"><summary>Gate activity per backbone, environment and preset (reused steps in percent, episodes with at least one reuse, success)</summary>

| Backbone and environment | Strict: reused %, episodes, success | Moderate | Aggressive |
|---|---|---|---|
| CogACT WidowX | 0.32, 11 of 200, 51.5 | 0.94, 26, 50.0 | 2.72, 41, 53.0 |
| CogACT Fractal | 0.01, 1 of 250, 66.8 | 0.11, 11, 67.2 | 0.57, 22, 66.8 |
| OpenVLA WidowX | 3.70, 51 of 200, 45.5 | 4.82, 68, 41.5 | 6.27, 72, 41.0 |
| OpenVLA Fractal | 2.28, 37 of 250, 36.0 | 2.50, 45, 36.4 | 3.88, 56, 35.2 |
| OpenVLA LIBERO Long | 3.61, 92 of 100, 51.0 | 5.58, 95, 51.0 | 10.14, 97, 50.0 |
| OpenVLA LIBERO Goal | 4.80, 69 of 100, 71.0 | 5.13, 87, 69.0 | 11.16, 92, 72.0 |
| OpenVLA LIBERO Object | 3.68, 86 of 100, 85.0 | 5.55, 97, 80.0 | 9.62, 100, 85.0 |
| OpenVLA LIBERO Spatial | 2.02, 63 of 100, 82.0 | 3.53, 85, 76.0 | 9.01, 93, 77.0 |
| SpatialVLA WidowX | 0.02, 2 of 200, 45.0 | 0.07, 6, 45.0 | 0.42, 20, 45.5 |
| SpatialVLA Fractal | 0.52, 17 of 250, 59.2 | 0.83, 44, 60.0 | 2.40, 85, 58.8 |
| CronusVLA WidowX | 0.05, 5 of 200, 36.0 | 0.33, 19, 34.5 | 1.38, 38, 36.5 |
| CronusVLA Fractal | 0.69, 20 of 250, 54.4 | 1.70, 41, 54.8 | 4.22, 80, 56.8 |
| UniVLA WidowX | 0.00, 0 of 200, 87.5 | 0.02, 1, 87.5 | 0.03, 1, 86.5 |
| UniVLA LIBERO Long | 0.02, 3 of 100, 86.0 | 0.02, 3, 86.0 | 0.06, 8, 86.0 |
| UniVLA LIBERO Goal | 0.00, 0 of 100, 93.0 | 0.00, 0, 93.0 | 0.00, 0, 92.0 |
| UniVLA LIBERO Object | 0.00, 0 of 100, 96.0 | 0.00, 0, 96.0 | 0.00, 0, 96.0 |
| UniVLA LIBERO Spatial | 0.00, 0 of 100, 95.0 | 0.00, 0, 95.0 | 0.00, 0, 95.0 |
| MiniVLA WidowX | 3.13, 41 of 200, 38.5 | 4.24, 56, 36.0 | 4.04, 65, 33.5 |
| SmolVLA LIBERO Long | 0.13, 26 of 100, 40.0 | 0.54, 64, 40.0 | 2.62, 92, 42.0 |
| SmolVLA LIBERO Goal | 0.13, 10 of 100, 77.0 | 0.46, 26, 76.0 | 2.17, 61, 79.0 |
| SmolVLA LIBERO Object | 0.07, 9 of 100, 89.0 | 0.22, 18, 89.0 | 1.42, 54, 90.0 |
| SmolVLA LIBERO Spatial | 0.02, 3 of 100, 78.0 | 0.25, 20, 78.0 | 1.37, 57, 76.0 |

</details>

The gates open on about a tenth of the steps at most (9 to 11 percent on OpenVLA LIBERO with the aggressive preset) and on well under 1 percent on CogACT Fractal, SpatialVLA WidowX and every UniVLA cell. On UniVLA Object, Spatial and WidowX strict no gate ever fired and the runs are identical to the original in success, steps and calls; on UniVLA Goal no gate fired either, and the runs differ from the original on one episode, which is run-to-run noise (Section 2.4).

### 1.5 Temporal fusion

The selector of Eq. (8) protects patches with motion, high entropy, high text-to-vision attention, or inside a dilated neighbourhood of those, and reuses the rest from the previous call up to a cap. The recorded arguments are identical on every backbone that recorded them:

| Parameter | motion-entropy | task-aware | conservative-adaptive |
|---|---|---|---|
| Motion threshold (patches above it are protected) | 0.01 | 0.01 | 0.01 |
| Entropy: fraction of highest-entropy patches protected | 0.15 | 0.15 | 0.15 |
| Task relevance: fraction of highest-attention patches protected | 0.2 | 0.2 | 0.2 |
| Dilation radius of the protected mask (patches) | 1 | 1 | 1 |
| Cap on the reusable fraction of patches | 0.5 | 0.5 | 0.25 |
| Keyframe interval (every k-th call recomputes everything) | 3 | 3 | 2 |
| Forced keyframe when frame motion exceeds | none | none | 0.03 |
| Text-to-vision attention of the preceding call collected | no | yes | no |

Fusion does not change the number of policy calls. Where it acts differs by backbone: CogACT fuses its projected visual tokens before the Llama decoder, SpatialVLA its 16 x 16 projected visual tokens before Gemma, UniVLA the discrete VQ codes before the language model; OpenVLA and MiniVLA fuse their 256 projected visual tokens (the 0.25 cap gives exactly 64 reused tokens under conservative-adaptive). Collecting the attention for the task-aware setting forces an SDPA decoder into eager attention, so on MiniVLA and UniVLA the task-aware row also changes the attention backend; on OpenVLA LIBERO the original already runs eager. CogACT exposes no text-to-vision attention where fusion runs, so its task-aware setting equals motion-entropy episode for episode.

<details markdown="1"><summary>Keyframe share and median reusable tokens per backbone, environment and setting</summary>

| Backbone and environment | motion-entropy: keyframes / calls, median tokens | task-aware | conservative-adaptive |
|---|---|---|---|
| CogACT WidowX | 0.34, 108 | same as motion-entropy | 0.93, 64 |
| CogACT Fractal | 0.34, 85 to 128 per task | same as motion-entropy | 0.91, 64 |
| OpenVLA WidowX | 0.34, 78 to 128 | 0.34, 46 to 91 | 0.93, 64 |
| OpenVLA Fractal | 0.34, 102 to 117 | 0.34, 47 to 62 | 0.82, 64 |
| OpenVLA LIBERO Long / Goal / Object / Spatial | 0.33 to 0.34, 124 to 128 | 0.33 to 0.34, 61 to 77 | 0.85 to 0.93, 64 |
| SpatialVLA WidowX | 0.34, 57 to 108 | 0.34, 26 to 54 | 0.98, 0 |
| SpatialVLA Fractal | 0.34, 69 to 118 | 0.34, 19 to 48 | 0.99, 0 |
| UniVLA WidowX | 0.37, 314 to 442 | not recorded (rerun) | 1.00, 0 |
| UniVLA LIBERO Long / Goal / Object / Spatial | 0.35 to 0.37, 300 to 312 | 0.35 to 0.36, 141 to 168 | 1.00, 0 |
| MiniVLA WidowX | 0.34, 107 | 0.34, 68 | 0.90, 64 |
| SmolVLA LIBERO (per suite) | not recorded, per-episode medians 19 to 23 | 3.5 to 5 | 16 on every episode |
| CronusVLA WidowX / Fractal | 106 / 112 fused patches per call, identical across the three settings | same | same |

</details>

Two consequences matter for reading the tables. Under conservative-adaptive the forced keyframe makes 82 to 93 percent of calls keyframes on OpenVLA, CogACT and MiniVLA, and on SpatialVLA and UniVLA every call is a keyframe with zero reused tokens, so those cells are the original policy under the fusion name. CronusVLA's three fusion settings are one run whose settings were not recorded.

### 1.6 Harness, checkpoints, software and hardware

**Episodes, caps and seeds.**

| Environment | Tasks | Episodes per task | Step cap |
|---|---|---|---|
| SimplerEnv WidowX | 4 (spoon on towel, carrot on plate, stack cube, eggplant in basket) | 50 | 60, and 120 for eggplant in basket |
| SimplerEnv Google Robot (Fractal) | 5 (close drawer, move near, open drawer, pick coke can, place apple in closed top drawer) | 50 | 80 |
| LIBERO Long, Goal, Object, Spatial | 10 each | 10 | 520, 300, 280, 220 |

Every configuration replays the same task instances with seed 42 plus the episode index (per-episode seeds are recorded on every backbone except the legacy CronusVLA runs). A failed episode runs to its cap, so 21,568 of the 21,568 failed episodes end exactly at the cap and average steps follows success. WidowX runs under the visual-matching protocol at 5 Hz control; LIBERO episodes settle for 10 no-op steps before the first call, and success is latched at the first step the goal predicate holds. Per pair: 200 episodes per WidowX configuration, 250 per Fractal configuration, 100 per LIBERO suite and configuration; 22 pairs x 14 configurations = 47,600 episodes.

**Checkpoints and per-backbone settings** (names as released; local paths omitted).

| Backbone | Checkpoint | Decoder pruned | Executed actions per call | Other recorded settings |
|---|---|---|---|---|
| CogACT | CogACT-Base (Llama-2-7B base) | 32 Llama layers | 1 | DDIM 10 steps, cfg scale 1.5; Fractal uses the Google Robot policy setup |
| OpenVLA | openvla-7b on SimplerEnv; openvla-7b-finetuned-libero-{spatial, object, goal, 10} on LIBERO (one per suite) | 32 Llama-2 layers | 1 | image resolution 224, no centre crop |
| SpatialVLA | spatialvla-4b-224-sft-bridge and -sft-fractal (PaliGemma2-3B) | 26 Gemma2 layers | 1 (native chunk 4 with action ensembling, temperature -0.8; queried every step) | attention: mixed, text decoder eager |
| CronusVLA | released checkpoint (training step 42,500) | 12-layer DiT action decoder | 1 | separate depth calibration pass, seed 10000 |
| UniVLA | UNIVLA_SIMPLER_BRIDGE_VIDEO_BS128_20K (WidowX), UNIVLA_LIBERO_VIDEO_BS192_8K (LIBERO), Emu3 vision tokenizer | 32 Emu3 layers | chunk of 5 (WidowX) or 10 (LIBERO) | attention SDPA |
| MiniVLA | minivla-vq-bridge-prismatic (prism-qwen25-extra-dinosiglip-224px, 0.5B) | 24 Qwen2.5 layers | 1 | image resolution 224, no centre crop |
| SmolVLA | LIBERO fine-tuned SmolVLA with SmolVLM2-500M-Instruct | 32 SmolLM2 layers (fixed indices) | 1 | lerobot 0.4.4, transformers 4.51.3 (legacy rows) |

**Attention backends and library versions.** CogACT, CronusVLA and MiniVLA run under transformers 4.47.0, where their Llama, Qwen2.5 and Qwen2.5 decoders resolve implicitly to SDPA (MiniVLA's own environment pins 4.40.1; the evaluation used 4.47.0). SpatialVLA's Gemma2 text decoder falls back to eager attention because of logit soft-capping, with SigLIP and ZoeDepth as separate components. SmolVLA calls PyTorch scaled_dot_product_attention explicitly. OpenVLA requests SDPA on SimplerEnv and its LIBERO wrapper defaults to eager; UniVLA requests SDPA. The transformers and torch versions of the OpenVLA and UniVLA runs are not recorded in the run files.

**GPU cards.** The runs were scheduled on a shared cluster whose compute nodes carry RTX 5090 cards, and latency is compared only within one backbone, one environment and one software stack. The run files record the card only for the runs below; every other run carries no GPU field.

| Runs | GPU recorded |
|---|---|
| CronusVLA WidowX guarded reuse moderate and aggressive (reruns) | RTX 5090, all 200 episodes each |
| UniVLA WidowX temporal fusion task-aware (rerun) | RTX 5090, all 200 episodes |
| SmolVLA guarded reuse and temporal fusion, all suites (re-implemented evaluator) | RTX 5090, all episodes |
| SmolVLA depth pruning 2 layers, Long and Goal (43 re-implemented episodes of 200) | RTX 5090 (33), RTX PRO 6000 (10) |
| SmolVLA depth pruning 4 layers, Long, Goal, Object (262 re-implemented episodes of 300) | RTX 5090 (186), RTX 6000 Ada (46), L40S (18), RTX A6000 (12) |

**Two implementations of SmolVLA.** SmolVLA's guarded reuse and temporal fusion cells, and part of its two- and four-layer depth cells on Long, Goal and Object, were completed with a re-implemented evaluator (tag `smolvla-reconstructed-sdpa-v1`) that is 3 to 7 percent faster per call than the legacy implementation even where the trick does nothing. Their latency values in Table II therefore mix an implementation difference with the trick, and those cells should be read through the paired success test alone. The three reruns above carry a GPU tag that their original rows lack, so their latency is not compared either.

**Latency definition.** Latency is wall-clock per environment step: the mean episode time divided by the mean episode length. The episode clock runs from reset to termination or cap, so it includes simulator stepping and rendering, the policy calls, the foveation blur, the reuse gate's signature computation, the fusion selector and, for the six backbones calibrated on the first frame, that unpruned first call. It excludes model loading, environment construction and CronusVLA's separate calibration pass. Every harness except the legacy CronusVLA runs also records the per-call model time, which Section 3 uses to separate model cost from environment cost.

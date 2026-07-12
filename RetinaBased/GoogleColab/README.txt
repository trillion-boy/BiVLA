Methodology Notes
=================

The base policy is OpenVLA. The question is whether we can preserve, or in some cases even improve, task performance if we stop feeding it a dense full-resolution view at every decision step. The repository tests that question in three conditions:

1. Vanilla OpenVLA, which sees the unmodified RGB image every step.
2. Foveated OpenVLA, which sees a log-polar compressed version of the image every step.
3. Retinotopic cached OpenVLA, which also works in log-polar space, but updates only selected visual regions and sometimes reuses the previous action instead of calling the model again.

The central motivation is practical. VLA policies are expensive mostly because they repeatedly process large visual inputs and run a heavy forward pass at every control step. Real robots, however, do not need to treat every pixel and every frame equally. Most manipulation scenes change slowly, and the most behaviorally relevant changes usually happen near the object of interest, not uniformly across the whole image. This repository is an attempt to operationalize that intuition in a concrete evaluation loop.


What is fixed, and what is allowed to change
--------------------------------------------

Several parts of the pipeline are held constant so that the comparison is meaningful.

- The underlying policy backbone is always OpenVLA.
- The model weights are not retrained or fine-tuned inside this repository.
- The language prompting format is fixed across all three variants.
- The same SimplerEnv tasks, camera viewpoint, episode splits, and robot control interface are used across conditions.

What changes is the input pathway and, in the retinal condition, the policy call schedule.

That distinction matters. The work is not claiming a new policy architecture. It is testing whether better input structuring and more selective inference can buy efficiency, and perhaps robustness, while leaving the policy itself intact.


Environment and benchmark setup
-------------------------------

All evaluations are run through `simple_eval.py`, which builds SimplerEnv tasks backed by the vendored `SimplerEnv/` subtree and its `ManiSkill2_real2sim` assets.

The evaluation uses four WidowX manipulation tasks:

- `widowx_put_eggplant_in_basket`
- `widowx_carrot_on_plate`
- `widowx_stack_cube`
- `widowx_spoon_on_towel`

Each task definition in `simple_eval.py` fixes:

- the SimplerEnv environment name
- the robot configuration
- the scene asset
- the observation camera
- the control frequency and simulation frequency
- the episode horizon
- the object episode range used for resets
- the robot's initial XY position when specified

The code evaluates 24 episodes by default. If episode IDs are not passed manually, it uses the task's configured object episode range and cycles through those episode IDs. For the current tasks that range is `[0, 24)`, which means the standard run covers episode IDs `0` through `23`.

Observations are requested in `rgbd` mode, but the policy itself is driven from the RGB image extracted by `get_image_from_maniskill2_obs_dict(...)` for the configured camera, which is `3rd_view_camera` in all current tasks.

Segmentation is also enabled in the environment. The script computes an `oracle_context` dictionary from segmentation masks and tracked object IDs, but the three policy wrappers in this repository do not actually consume that information yet. The same is true for the assembled proprioceptive state and phase info: the evaluator computes them and passes them in, but the present OpenVLA wrappers do not use them to change the action.

This is worth stating plainly because it keeps the method honest: right now, the comparison is driven by image transformation and inference reuse, not by a richer multimodal controller.


Why there is a real-image overlay in the environment
----------------------------------------------------

Each task configuration points to an RGB overlay image inside `ManiSkill2_real2sim/data/real_inpainting/...`. During environment construction, if that overlay path exists, it is attached to the relevant camera.

The practical role of that overlay is to make the rendered scene visually closer to the real backgrounds used in the Bridge-style evaluation setup. In other words, the benchmark is not a clean synthetic tabletop world. It intentionally pushes the policy toward a more realistic visual regime. That choice makes the visual preprocessing question more meaningful, because any compression scheme has to survive clutter, texture, lighting variation, and background detail rather than a toy rendering.


Policy interface and action decoding
------------------------------------

All three policy classes live in `openvla_inference.py`.

The prompt sent to the model is always:

In: What action should the robot take to {instruction}?
Out:

The evaluator obtains the instruction from `env.get_language_instruction()`. If the environment changes the instruction mid-episode, the code updates the instruction and resets the policy wrapper's internal cache.

OpenVLA returns a raw action vector. The wrapper decodes it as follows:

- elements `0:3` become the Cartesian translation command, stored as `world_vector`
- elements `3:6` are interpreted as Euler angles, converted to axis-angle form, and stored as `rot_axangle`
- element `6` becomes the gripper command, thresholded to `+1.0` if above `0.5`, otherwise `-1.0`
- `terminate_episode` is always set to `0.0` in this code

The evaluator then flattens that action into the control vector expected by the environment:

- 3 translation values
- 3 axis-angle rotation values
- 1 gripper value

So, for the purpose of this work, the comparison point is identical motor output format under different visual input schedules.


Condition 1: Vanilla OpenVLA
----------------------------

The vanilla wrapper does almost nothing to the image. `prepare_image(...)` simply returns the input RGB frame as an unsigned 8-bit array. The model is queried on every control step. There is no caching, no compression, and no action reuse.

This is the reference condition. Any speed or performance change in the other two settings has to be understood relative to this baseline.


Condition 2: Foveated OpenVLA
-----------------------------

The second condition keeps the same model and the same per-step query schedule, but changes the image representation before it reaches the processor.

The transformation works like this:

1. The RGB frame is converted into log-polar coordinates with OpenCV's `warpPolar(..., WARP_POLAR_LOG)`.
2. A uniformly spaced sampling grid is drawn over that log-polar image.
3. The number of sampled rows and columns is scaled so that the retained area is approximately the requested keep ratio.
4. The sampled image is resized back to the original image size.
5. The result is mapped back to Cartesian image space with the inverse log-polar transform.

The default keep percentage is `20%`, which means the code keeps about one fifth of the visual sample density. Because the row and column counts are both scaled by `sqrt(keep_ratio)`, the retained 2D area is roughly the desired proportion.

The point of doing this in log-polar space is straightforward: it places relatively more representational emphasis near the center and progressively less in the periphery. That mimics the basic logic of biological foveation without needing a learned attention module.

Nothing else changes in this condition. The model still runs every step.


Condition 3: Retinotopic cached OpenVLA
---------------------------------------

The third condition is where the main methodological contribution sits.

It starts from the same log-polar foveation idea, but then adds two more ideas:

1. keep a cached log-polar representation across time
2. avoid calling the model when the scene has not changed enough to justify a new forward pass

The wrapper is implemented in `RetinotopicCachedOpenVLAInference`.

Step A: build the current log-polar observation
-----------------------------------------------

For each incoming frame, the code computes:

- the full log-polar image
- a sampled-and-resized log-polar image using the chosen keep ratio
- the transform metadata needed to map back to Cartesian space

The wrapper also stores the previous log-polar frame so it can measure temporal change.

Step B: estimate motion in three retinotopic bands
--------------------------------------------------

The code compares the current and previous log-polar frames with a normalized absolute difference. It then averages that motion signal over three horizontal bands in log-polar space:

- fovea
- mid-periphery
- outer periphery

By default, the band boundaries are controlled by:

- `fovea_fraction = 0.22`
- `mid_fraction = 0.55`

That means the first 22 percent of rows are treated as the foveal region, the next band runs up to 55 percent, and the remainder is treated as outer periphery.

Step C: decide between full refresh and partial refresh
-------------------------------------------------------

A full refresh happens when one of the following is true:

- there is no cache yet
- global motion is at least `0.060`
- foveal motion exceeds twice the foveal threshold

With the default parameters, the doubled foveal trigger corresponds to `2 x 0.015 = 0.030`.

If none of those conditions hold, the wrapper performs a partial update of the cached log-polar image:

- the fovea is always refreshed
- the mid band is refreshed every `2` steps or when its motion exceeds `0.025`
- the outer band is refreshed every `4` steps or when its motion exceeds `0.040`

After refreshing the selected bands, the cached log-polar image is mapped back into Cartesian image space and that reconstructed frame is what the model would see if a forward pass is requested.

This is an important detail. The model is not being modified to read log-polar tensors directly. The method edits the image stream upstream and then hands OpenVLA a normal image-shaped input.

Step D: decide whether the model needs to run
---------------------------------------------

Even after the visual cache is updated, the wrapper may skip a model call and reuse the previous raw action.

The model is forced to run when:

- there is no previous action yet
- the current step triggered a full visual refresh
- foveal motion is at least `max(0.018, 0.015)`, which is `0.018` under the defaults
- the number of consecutive reused steps has reached the reuse limit, which is `2`
- the norm of the previous action is too large to trust reuse, using the code's metric:
  translation norm + 0.5 x rotation norm

The default maximum allowed reuse-action norm is `0.08`.

If none of those conditions are met, the wrapper reuses the previous raw OpenVLA action instead of running another forward pass.

This is the key efficiency move in the repository. The retinal condition reduces cost in two ways at once:

- it reduces how much of the image is freshly updated
- it reduces how often the policy network itself is queried


Episode loop and evaluation protocol
------------------------------------

For each episode, `simple_eval.py` does the following:

1. Build the environment from the selected task config.
2. Reset the scene with the chosen episode ID and robot initialization.
3. Read the language instruction from the environment.
4. Reset the model wrapper.
5. Repeatedly extract the RGB image, produce an action, and step the environment until success, truncation, or the task's step limit.

During the loop, the evaluator also tracks:

- whether the source object has ever been grasped
- elapsed wall-clock time
- per-episode step count
- final environment info

If `--save-video` is passed, the script saves a GIF composed from the prepared images seen by the policy wrapper. For the retinal model, that means the saved frames can reflect the cached and partially refreshed representation rather than the raw camera frame.

At the end of a run, the script writes a JSON summary under:

- `results/openvla/...`
- `results/openvla_foveated/...`
- `results/openvla_retina/...`

depending on the selected model.


What is actually being measured
-------------------------------

The evaluator records standard task-level outcomes:

- success rate
- grasp rate
- average number of steps
- average elapsed time

For the retinal condition it also records internal efficiency statistics, including:

- number of policy steps
- number of model calls
- number of reused actions
- counts of full and partial refreshes
- counts of fovea, mid, and outer refreshes
- mean refresh ratio
- mean global motion
- model call rate
- estimated speedup versus a one-call-per-step baseline

That extra logging matters because otherwise a faster runtime would be hard to interpret. With those counters, the repository can distinguish between "the model got faster because episodes ended sooner" and "the model got faster because fewer forward passes were executed."


What the saved results currently show
-------------------------------------

Using the JSON files already present in `results/`, the broad picture is:

- vanilla OpenVLA is the weakest condition on these four tasks
- simple foveation helps substantially on all four tasks
- the retinal cached version cuts model calls roughly in half and noticeably reduces wall-clock time
- retinal caching does not dominate the simpler foveated baseline on success rate, but it does deliver a better efficiency/performance trade-off than the vanilla baseline

Stored task summaries:

- `openvla`
  `widowx_carrot_on_plate`: success `0.0000`, grasp `0.2083`, avg elapsed `11.01s`
  `widowx_put_eggplant_in_basket`: success `0.0000`, grasp `0.1250`, avg elapsed `22.73s`
  `widowx_spoon_on_towel`: success `0.0417`, grasp `0.2083`, avg elapsed `10.88s`
  `widowx_stack_cube`: success `0.0833`, grasp `0.1667`, avg elapsed `10.27s`

- `openvla_foveated`
  `widowx_carrot_on_plate`: success `0.2083`, grasp `0.3333`, avg elapsed `10.93s`
  `widowx_put_eggplant_in_basket`: success `0.4167`, grasp `0.7917`, avg elapsed `16.93s`
  `widowx_spoon_on_towel`: success `0.4583`, grasp `0.7083`, avg elapsed `9.36s`
  `widowx_stack_cube`: success `0.2083`, grasp `0.6250`, avg elapsed `10.86s`

- `openvla_retina`
  `widowx_carrot_on_plate`: success `0.0417`, grasp `0.2083`, avg elapsed `7.33s`, model call rate `0.4684`, estimated speedup `2.2755x`
  `widowx_put_eggplant_in_basket`: success `0.2917`, grasp `0.6250`, avg elapsed `12.79s`, model call rate `0.5075`, estimated speedup `2.0555x`
  `widowx_spoon_on_towel`: success `0.1667`, grasp `0.3333`, avg elapsed `7.20s`, model call rate `0.4841`, estimated speedup `2.1868x`
  `widowx_stack_cube`: success `0.0833`, grasp `0.2083`, avg elapsed `6.85s`, model call rate `0.4392`, estimated speedup `2.4401x`

To my reading, the main lesson is not that the retinal scheme is universally best in absolute task success. It is that selective visual updating is strong enough to keep the policy functional while removing a large fraction of expensive model evaluations. That is the methodological point this repository demonstrates most clearly.


How to run the evaluations
--------------------------

There are now two supported ways to run this project:

1. the local shell launcher, `run_experiment.sh`
2. the Colab notebook, `BiVLA_colab.ipynb`

Both paths execute the same evaluation script, `simple_eval.py`, and the same three model variants:

- `openvla`
- `openvla_foveated`
- `openvla_retina`

Local launcher
--------------

The main local launcher is `run_experiment.sh`. It sources `configs/paths.sh`, activates the configured conda environment, and runs `simple_eval.py`.

Typical examples:

`./run_experiment.sh --model openvla --task widowx_stack_cube`

`./run_experiment.sh --model openvla_foveated --task widowx_spoon_on_towel --foveated-keep-percent 20`

`./run_experiment.sh --model openvla_retina --task widowx_put_eggplant_in_basket --retina-max-action-reuse 2`

By default, the launcher uses:

- conda env: `bivla`
- model path: `openvla/openvla-7b`
- unnormalization key: `bridge_orig`
- device: `cuda`

If `xvfb-run` is available, the script uses it for offscreen rendering.


Google Colab path
-----------------

The Colab notebook version is `BiVLA_colab.ipynb`. It is aligned with the same repository logic, but it sets the environment up in the way Colab expects:

- it installs system packages with `apt`
- it installs Python dependencies with `pip`
- it exports the same core environment variables used by the project
- it starts `Xvfb` manually for headless rendering
- it runs `simple_eval.py` directly through Python rather than through the local shell launcher

The notebook includes:

- a repository setup cell
- a full dependency-install cell
- headless rendering setup for SAPIEN / SimplerEnv
- optional Hugging Face login for gated model access
- a smoke test that runs all three model variants on one task
- a full evaluation cell that runs the complete model-task matrix

The default notebook settings are intentionally kept close to the repository defaults:

- model path: `openvla/openvla-7b`
- unnormalization key: `bridge_orig`
- device: `cuda`
- default episode count: `24`

The one practical difference is output location. The notebook writes its runs under `results_colab/` by default so that Colab-produced outputs do not overwrite the existing repository `results/` directory unless you choose to point them there.

There is one hardware caveat that matters. The notebook preserves the repository's direct OpenVLA loading path and therefore uses CUDA `bfloat16` when a GPU is present. In practice, that means Colab sessions with an `L4` or `A100` are a safer fit than older `T4` sessions, which can be memory-constrained for the exact `openvla/openvla-7b` setup.


What this work is, in one sentence
----------------------------------

This repository is an evaluation of whether OpenVLA can be made cheaper and, in some cases, behaviorally stronger by replacing dense frame-by-frame vision with foveated, retinotopically refreshed visual input and selective action reuse.


What this work is not
---------------------

It is not:

- a new trained VLA backbone
- a fine-tuning pipeline
- a learned attention mechanism
- a claim that oracle segmentation or proprioception is driving the current results

Those pieces are either fixed, unused, or outside the scope of the current code.


Final interpretation
--------------------

The repository keeps OpenVLA's policy machinery intact and moves the experimental pressure onto the observation stream. First, it asks whether a spatially non-uniform image representation is enough. Then it asks whether temporal persistence can be exploited so the policy does not have to reconsider the entire scene every time the control loop ticks. The answer, at least in these stored runs, appears to be yes: a surprising amount of manipulation competence survives under aggressive visual economy, and some tasks even improve relative to the raw baseline.

That is the methodological core of the work.

What this folder is
-------------------

This `Shareable/` directory is a trimmed, self-contained reproduction bundle for the BiVLA evaluation in this repository.

It keeps only the pieces required to run the current four WidowX tasks under the three OpenVLA-based policy conditions:

- `openvla`
- `openvla_foveated`
- `openvla_retina`

The goal here is not to mirror the entire parent repository. The goal is to make the runnable surface small, explicit, and portable.


What is included
----------------

At the top level:

- `simple_eval.py`
- `openvla_inference.py`
- `run_experiment.sh`
- `configs/nvidia_icd_egl.json`

Inside `SimplerEnv/`:

- the small `simpler_env` helper package used by the evaluator
- the `ManiSkill2_real2sim` Python package
- only the asset subset required by the four configured WidowX tasks

That asset subset includes:

- `bridge_table_1_v1` and `bridge_table_1_v2` stage files
- `bridge_real_eval_1.png` and `bridge_sink.png` overlay images
- the model-info JSON files used by the Bridge task environments
- the object models needed by the spoon, carrot, cube, and eggplant tasks

Large unrelated assets from the full parent repository were intentionally left out.


Method in plain language
------------------------

This package evaluates one backbone, OpenVLA, under three observation / inference schedules.

1. `openvla`
   The model sees the unmodified RGB image and is called every step.

2. `openvla_foveated`
   The input image is converted to log-polar space, sampled more aggressively in the periphery, mapped back to image space, and then passed to OpenVLA every step.

3. `openvla_retina`
   The model still uses the same log-polar idea, but it also caches the transformed image across time, refreshes only selected retinotopic bands when needed, and sometimes reuses the previous action instead of running a new forward pass.

The policy weights are not retrained here. The experiment changes the visual input pathway and, in the retinal condition, the timing of model calls.


Why this shareable package is smaller than the full repo
--------------------------------------------------------

The original repository contains more than this evaluation needs:

- extra SimplerEnv content
- extra custom assets
- extra environments unrelated to the four active WidowX tasks
- local result files and convenience scripts

This shareable version keeps the exact evaluation code path but narrows the asset and setup surface to the minimum practical subset for the active benchmark.


Single script entry point
-------------------------

Everything is driven by one script:

`./run_experiment.sh`

It contains:

- virtualenv creation
- Python dependency installation
- optional `apt` installation of system dependencies
- optional Hugging Face login if `HF_TOKEN` is set
- environment-variable setup
- automatic `Xvfb` startup for headless rendering when available
- `smoke`, `full`, and custom `run` entry points


Setup
-----

Basic setup:

`./run_experiment.sh setup`

If you also want the script to install system libraries through `apt`, run:

`./run_experiment.sh setup --with-apt`

Notes:

- `--with-apt` only works when `apt-get` is available and you have root or `sudo`.
- the script creates a local virtualenv in `./.venv` by default
- if your OpenVLA checkpoint requires authentication, export `HF_TOKEN` before running setup or evaluation
- if you skip `setup`, the `smoke`, `full`, and `run` commands will still bootstrap the Python environment automatically, but they will not install `apt` packages for you

Useful environment variables:

- `PYTHON_BIN`
- `VENV_DIR`
- `RESULTS_ROOT`
- `OPENVLA_MODEL_PATH`
- `OPENVLA_UNNORM_KEY`
- `DEVICE`
- `HF_TOKEN`


Run commands
------------

Minimal smoke test:

`./run_experiment.sh smoke`

This runs all three model variants on `widowx_spoon_on_towel` with `1` episode each.

Full benchmark:

`./run_experiment.sh full`

This runs:

- 4 tasks
- 3 model variants
- 24 episodes per task-model pair

Custom run:

`./run_experiment.sh run --model openvla_retina --task widowx_put_eggplant_in_basket --n-episodes 4`

The `run` mode passes arguments straight through to `simple_eval.py`, so any evaluator flags supported there can still be used.


Outputs
-------

This shareable package writes results under:

`results_shareable/`

The script mirrors the parent repository's output convention:

- `results_shareable/openvla/...`
- `results_shareable/openvla_foveated/...`
- `results_shareable/openvla_retina/...`

Each run writes a JSON summary named:

`results_<task>.json`


Hardware note
-------------

This package preserves the repository's direct OpenVLA loading path. On CUDA devices, the code uses `bfloat16` in the same way as the parent project.

Practically, that means:

- a modern GPU is strongly preferred
- `L4` or `A100` class hardware is safer than `T4` class hardware
- CPU-only execution is possible in principle but will be very slow for the actual benchmark


What stayed aligned with the parent repository
----------------------------------------------

This shareable pack keeps the same:

- task definitions
- OpenVLA prompt format
- action decoding logic
- foveation transform
- retinal refresh rules
- model-call reuse logic
- result JSON structure

So the behavior is meant to match the parent evaluation path, not a simplified reimplementation.


Recommended order
-----------------

1. Run `./run_experiment.sh setup` first.
2. Run `./run_experiment.sh smoke`.
3. Only after the smoke test works, run `./run_experiment.sh full`.

That is the safest way to confirm that model access, rendering, assets, and the local Python environment are all in place before the expensive benchmark starts.

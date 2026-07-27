# OpenVLA on LIBERO — Colab setup

Copy-paste cells for a **completely fresh runtime** (nothing cloned, nothing
installed). Roughly 20–30 min of setup, most of it the ~15GB checkpoint
download.

## Why OpenVLA for LIBERO

SpatialVLA reports LIBERO numbers but never released those weights — the
`IPEC-COMMUNITY` org only ships `224-pt`, `mix-224-pt`, `sft-bridge`,
`sft-fractal`. LIBERO cannot be run zero-shot (success collapses to ~0
without a LIBERO fine-tune), so a public LIBERO checkpoint is mandatory.
OpenVLA publishes all four:

- `openvla/openvla-7b-finetuned-libero-spatial`
- `openvla/openvla-7b-finetuned-libero-object`
- `openvla/openvla-7b-finetuned-libero-goal`
- `openvla/openvla-7b-finetuned-libero-10`

OpenVLA is also the standard LIBERO baseline that the SpatialVLA and UniVLA
papers both report against, so it is the most defensible backbone here.
Reference success rates: spatial 84.7%, object 88.4%, goal 79.2%, long 53.7%.

**Each checkpoint is trained on one suite only.** Always pair the checkpoint
with its matching `--task-suite`; crossing them measures nothing.

## 1. Clone BiVLA

```bash
%cd /content
!git clone -b claude/serene-davinci-sy33re https://github.com/trillion-boy/bivla.git BiVLA
%cd /content/BiVLA
!git log -1 --oneline
```

## 2. System packages (headless MuJoCo rendering)

```bash
!apt-get -qq update && apt-get -qq install -y libgl1-mesa-dri libegl1 libglvnd0 libgles2
```

## 3. Python packages

`libero` pins `robosuite==1.4.0`, which calls `mujoco.mj_fullM()` with the
pre-3.10 argument order; mujoco 3.10 changed that signature, so any env step
dies with `TypeError: mj_fullM(): incompatible function arguments`. Verified
across versions: fine through 3.9.0, broken from 3.10.0.

transformers must be pinned to exactly what OpenVLA documents. Leaving
Colab's preinstalled 5.x in place fails immediately with
`ImportError: cannot import name 'AutoModelForVision2Seq'` — that auto class
was removed in transformers 5, and OpenVLA's remote code
(`modeling_prismatic.py`) targets the 4.40 API. `timm==0.9.10` is equally
non-negotiable: the Prismatic vision backbone is built against that exact
timm version. This pinned trio imports cleanly alongside numpy 2.x
(verified: transformers 4.40.1 / timm 0.9.10 / tokenizers 0.19.1 /
numpy 2.5.1).

```bash
!pip install -q libero
!pip install -q "mujoco==3.9.0"
!pip install -q "transformers==4.40.1" "timm==0.9.10" "tokenizers==0.19.1" accelerate
```

## 4. Restart the runtime — required

Step 3 swapped `mujoco` under an already-imported module.
**Runtime → Restart session**, then continue at step 5. Do not re-run 1–3.

## 5. LIBERO config + 3D assets

`eval_libero.py` repairs a missing/truncated `~/.libero/config.yaml` itself,
but the assets still need one env build to download.

```python
import os, yaml, importlib.util

spec = importlib.util.find_spec("libero")
pkg_root = os.path.join(os.path.dirname(spec.origin), "libero")
os.makedirs(os.path.expanduser("~/.libero"), exist_ok=True)
with open(os.path.expanduser("~/.libero/config.yaml"), "w") as f:
    yaml.dump({
        "benchmark_root": pkg_root,
        "bddl_files": os.path.join(pkg_root, "bddl_files"),
        "init_states": os.path.join(pkg_root, "init_files"),
        "datasets": os.path.join(pkg_root, "..", "datasets"),
        "assets": os.path.join(pkg_root, "assets"),
    }, f)
os.environ["MUJOCO_GL"] = "egl"

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
bm = benchmark.get_benchmark_dict()["libero_spatial"]()
task = bm.get_task(0)
bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=128, camera_widths=128)
env.reset()
print("LIBERO env OK — assets downloaded")
env.close()
```

## 6. Download the checkpoint

One checkpoint per suite; start with `libero_spatial`.

```python
from huggingface_hub import snapshot_download
CKPT = snapshot_download(
    repo_id="openvla/openvla-7b-finetuned-libero-spatial",
    local_dir="/content/openvla-libero-spatial",
)
print(CKPT)
```

Check which action statistics it carries (the eval picks it automatically
when there is only one, which is the normal case):

```python
import json, os
cfg = json.load(open("/content/openvla-libero-spatial/config.json"))
print("norm_stats keys:", list(cfg.get("norm_stats", {}).keys()))
```

## 7. Smoke test — one task, one trial

```bash
%cd /content/BiVLA/adaptive_sparse_vla
!python eval_libero.py \
  --backbone openvla \
  --model-path /content/openvla-libero-spatial \
  --task-suite libero_spatial --task-ids 0 --n-trials-per-task 1 \
  --save-video --output-dir /content/bivla_eval_libero
```

Healthy output:

- `[openvla] unnorm_key=libero_spatial_no_noops ...`
- `trial 0: building env ... reset ... ready (Ns)`
- `[heartbeat] call N ...` every 5 calls
- `[debug] ... chunk_shape=(1, 7) dim_absmax=[...] first_row=[...]`

The gripper column (last value) must be exactly `-1.0` or `1.0`. Sensible
translation magnitudes are roughly ≤1.0. `libero_spatial` task 0 is easy, so
a single trial should have a real chance of `SUCCESS`.

Watch it:

```python
from IPython.display import Image as IPImage
import glob
IPImage(sorted(glob.glob("/content/bivla_eval_libero/*.gif"))[-1])
```

### If the run dies silently during `building env`

That is a native segfault from MuJoCo's EGL context colliding with CUDA,
already initialized by the model — not a Python error, so there is no
traceback. It was observed identically with UniVLA and OpenVLA on an A100,
while building the very same env in a fresh process (no model loaded)
succeeded, which is what identifies it as an ordering problem rather than a
broken EGL install.

`eval_libero.py` now creates the MuJoCo GL context *before* the policy
touches CUDA (look for `[env] MuJoCo GL context pre-initialized before
CUDA`), which is the standard fix for this ordering bug. If a run still dies
there, fall back to CPU rendering:

```bash
!apt-get -qq install -y libosmesa6-dev
!pip install -q PyOpenGL PyOpenGL_accelerate
```

then add `--mujoco-gl osmesa` to every eval command. Slower to render, but it
removes the GPU contention entirely.

## 8. The experiment runs

OpenVLA predicts one action per forward pass, so there is no chunk to
truncate — the efficiency intervention is `--action-repeat N`, which executes
each action N times open-loop and cuts forward passes by N×. `--exec-chunk`
is the equivalent knob for chunked policies (UniVLA), and reporting both
under one "fewer forward passes" axis is exactly the architecture-dependence
comparison.

```bash
%cd /content/BiVLA/adaptive_sparse_vla
CKPT=/content/openvla-libero-spatial
SUITE=libero_spatial
OUT=/content/bivla_eval_libero

# baseline
!python eval_libero.py --backbone openvla --model-path $CKPT \
  --task-suite $SUITE --n-trials-per-task 10 --output-dir $OUT

# efficiency: 2x fewer forward passes
!python eval_libero.py --backbone openvla --model-path $CKPT \
  --task-suite $SUITE --n-trials-per-task 10 \
  --action-repeat 2 --output-dir $OUT

# foveation, log-polar
!python eval_libero.py --backbone openvla --model-path $CKPT \
  --task-suite $SUITE --n-trials-per-task 10 \
  --foveate --foveate-mode logpolar --foveate-keep-percent 20 --output-dir $OUT

# foveation, blur
!python eval_libero.py --backbone openvla --model-path $CKPT \
  --task-suite $SUITE --n-trials-per-task 10 \
  --foveate --foveate-mode blur --foveate-keep-percent 20 --output-dir $OUT
```

Each run writes `summary_<suite>_<timestamp>.json` to `--output-dir` with
`backbone`, `checkpoint`, `action_repeat`, `exec_chunk`, the foveation
settings, per-episode results, mean inference latency, and the success rate.

Budget: a 10-task × 10-trial suite is 100 episodes of up to ~230 steps. Start
with `--n-trials-per-task 3` to get the shape of the result before spending
hours on the full grid.

To cover more suites later, swap both the checkpoint and the suite together:

| `--task-suite` | checkpoint |
|---|---|
| `libero_spatial` | `openvla/openvla-7b-finetuned-libero-spatial` |
| `libero_object` | `openvla/openvla-7b-finetuned-libero-object` |
| `libero_goal` | `openvla/openvla-7b-finetuned-libero-goal` |
| `libero_10` | `openvla/openvla-7b-finetuned-libero-10` |

## Notes carried over from the UniVLA work

The harness is shared, so these already apply:

- `~/.libero/config.yaml` is validated and repaired automatically. The
  original failure was a truncated write losing `init_states` (it sorts last
  alphabetically).
- Checkpoint paths are checked before the model loads, so a typo fails in
  seconds instead of after a multi-minute load.
- `MUJOCO_GL` defaults to `egl` inside the script; `--mujoco-gl` overrides.
- Foveation (`foveation.py`) is pixel-identical to the SimplerEnv runs, so
  the LIBERO and SimplerEnv foveation results are directly comparable.

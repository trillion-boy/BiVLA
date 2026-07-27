# UniVLA on LIBERO — Colab setup

Run this in Colab (or any environment with real internet access — this
sandbox's network policy blocks huggingface.co, so none of this could be
verified end-to-end there; everything up to the HF-gated steps was verified
against the real `libero` package). Run in the same environment/session
where your UniVLA SimplerEnv eval already works (same `bivla` conda env or
equivalent) — this only adds LIBERO-specific packages on top.

## 0. Clone BiVLA (the branch with this code)

`eval_libero.py` / `inference_libero.py` are not on `main` yet — they're on
the `claude/serene-davinci-sy33re` branch. If you already have a BiVLA clone
in this Colab from earlier SimplerEnv runs, just `git fetch` + `checkout`
that branch inside it instead of re-cloning:

```bash
%cd /content
!git clone -b claude/serene-davinci-sy33re https://github.com/trillion-boy/bivla.git BiVLA
%cd /content/BiVLA
!git log -1 --oneline   # sanity check: should show the LIBERO-harness commit
```

If `trillion-boy/bivla` is private and the plain clone above fails with an
auth error, use whatever method you already used to get BiVLA into Colab for
the SimplerEnv runs (a personal access token in the URL, an SSH deploy key,
or just re-running your existing clone cell and then `git checkout
claude/serene-davinci-sy33re` inside it).

If you already have `/content/BiVLA` cloned on a different branch:

```bash
%cd /content/BiVLA
!git fetch origin claude/serene-davinci-sy33re
!git checkout claude/serene-davinci-sy33re
!git pull
```

## 1. System packages

```bash
!apt-get -qq update && apt-get -qq install -y libgl1-mesa-dri libegl1 libglvnd0 libgles2
```

## 2. LIBERO simulator

`pip install libero` only requires `mujoco>=3.0.0`, which today resolves to
the latest 3.10.0 -- but `robosuite==1.4.0` (pinned by `libero`) calls
`mujoco.mj_fullM()` using the pre-3.10 argument order `(model, dst, M)`.
Mujoco 3.10.0 changed that signature to `(model, data, dst)`, so anything
that calls a controller (i.e. any real env step/reset) crashes with
`TypeError: mj_fullM(): incompatible function arguments`. Confirmed by
testing the signature across versions directly: unchanged through 3.9.0,
broken starting at 3.10.0. Pin it down one patch line:

```bash
!pip install -q libero
!pip install -q "mujoco==3.9.0"
```

If Python already had `mujoco` imported earlier in this session (it will
have been, once you've run any cell that touches `libero.libero.envs`),
restart the runtime after this before importing anything LIBERO-related
again — the old 3.10.0 module stays loaded in memory otherwise.

This pulls in `robosuite==1.4.0`, `mujoco`, `bddl`, `robomimic` — the real
LIBERO benchmark package (published by the original LIBERO authors via
HuggingFace's `lerobot-libero` repackaging), not a stand-in.

## 3. Pre-seed the LIBERO config (skips an interactive prompt on first import)

```python
import os, yaml, importlib.util

spec = importlib.util.find_spec("libero")
pkg_root = os.path.join(os.path.dirname(spec.origin), "libero")
config = {
    "benchmark_root": pkg_root,
    "bddl_files": os.path.join(pkg_root, "bddl_files"),
    "init_states": os.path.join(pkg_root, "init_files"),
    "datasets": os.path.join(pkg_root, "..", "datasets"),
    "assets": os.path.join(pkg_root, "assets"),
}
os.makedirs(os.path.expanduser("~/.libero"), exist_ok=True)
with open(os.path.expanduser("~/.libero/config.yaml"), "w") as f:
    yaml.dump(config, f)

os.environ["MUJOCO_GL"] = "egl"
```

## 4. Trigger the LIBERO 3D asset download

The `libero` pip package ships task definitions (bddl files) but not the 3D
scene/object assets — those download from HF (`jadechoghari/libero-assets`)
the first time an environment is actually built:

```python
from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv
from libero.libero import get_libero_path

bm = benchmark.get_benchmark_dict()["libero_goal"]()
task = bm.get_task(0)
bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
env = OffScreenRenderEnv(bddl_file_name=bddl_file, camera_heights=128, camera_widths=128)
env.reset()
print("LIBERO env OK — assets downloaded")
env.close()
```

## 5. Download the UniVLA LIBERO checkpoint

This is the real, public checkpoint from the UniVLA authors (baaivision/UniVLA,
ICLR 2026) — **not** the unrelated OpenDriveLab/UniVLA project of the same
name. Confirmed against this repo's own code: the checkpoint name
`UNIVLA_LIBERO_IMG_BS192_8K` already appears in
`UniVLA/models/inference/inference_action.py`.

```python
from huggingface_hub import list_repo_files, snapshot_download

# Check the real folder names first — I couldn't browse huggingface.co from
# the sandboxed session this was written in, so verify before assuming the
# exact subfolder names below.
files = list_repo_files("Yuqi1997/UniVLA")
print("\n".join(files))
```

Then, adjusting the `allow_patterns` if the printed listing differs:

```python
ckpt_dir = snapshot_download(
    repo_id="Yuqi1997/UniVLA",
    allow_patterns=["UNIVLA_LIBERO_IMG_BS192_8K/*"],
    local_dir="/content/UNIVLA_LIBERO_IMG_BS192_8K",
)
print(ckpt_dir)
```

Note there is no separate FAST-tokenizer folder in this repo at all (checked
the real file listing — only checkpoint folders, each with the usual
config.json/safetensors/tokenizer files). That's not a gap: read
`UniVLA/pretrain/fast_bridge_t5_s50/processing_action_tokenizer.py` in this
repo and it's a `UniversalActionProcessor` — a DCT+BPE action tokenizer with
`action_dim`/`time_horizon` left as `None` in its config, filled in at
`decode()` call time rather than baked in per-embodiment. That is exactly
the "FAST+" design (Physical Intelligence's action tokenizer, explicitly
built to be reused as a black box across different robots/action spaces
without refitting). So use the **same local folder already sitting at
`UniVLA/pretrain/fast_bridge_t5_s50`** — no separate download needed:

```bash
--fast-path /content/BiVLA/UniVLA/pretrain/fast_bridge_t5_s50
```

This is a reasonably confident inference, not a 100%-verified fact (I
couldn't load the actual checkpoint to confirm the token IDs line up). The
tell if it's wrong: the model would produce actions that don't correspond to
anything sensible even on the very first baseline run (not just "low success
rate" — actually garbled, structureless motion). If you see that, the
LIBERO checkpoint was likely trained with a separately-fit FAST vocabulary
that isn't in this HF repo, and it's worth asking whoever has access to the
original training cluster (`/share/project/yuqi.wang/...`) for the exact
`fast_path` they used.

## 5.5. The vision tokenizer is a SEPARATE checkpoint, not the LIBERO folder

`--vq-hub` (also used as `vision_hub` internally) must point at the frozen
Emu3 vision tokenizer -- the same `BAAI/Emu3-VisionTokenizer` already used
for the Bridge/SimplerEnv baseline (see `README.md` section 4.4: it's
downloaded once to `pretrain/Emu3-VisionTokenizer` and shared across every
UniVLA checkpoint, since it's a general-purpose frozen image tokenizer, not
embodiment-specific). It is **not** part of the
`UNIVLA_LIBERO_IMG_BS192_8K` folder -- that folder only has the LLM
(Emu3MoE) weights and its own text tokenizer. Passing the LIBERO checkpoint
path as `--vq-hub` fails with
`OSError: Can't load image processor for ... no preprocessor_config.json`,
because the LLM checkpoint folder genuinely has no vision-tokenizer files.

If you already downloaded `Emu3-VisionTokenizer` for the SimplerEnv/Bridge
runs in this same Colab, just reuse that local path. Otherwise:

```python
from huggingface_hub import snapshot_download
vision_dir = snapshot_download(
    repo_id="BAAI/Emu3-VisionTokenizer",
    local_dir="/content/BiVLA/pretrain/Emu3-VisionTokenizer",
)
print(vision_dir)
```

## 6. Run the eval

```bash
%cd /content/BiVLA/adaptive_sparse_vla

!python eval_libero.py \
  --emu-hub /content/UNIVLA_LIBERO_IMG_BS192_8K/UNIVLA_LIBERO_IMG_BS192_8K \
  --vq-hub /content/BiVLA/pretrain/Emu3-VisionTokenizer \
  --fast-path /content/BiVLA/UniVLA/pretrain/fast_bridge_t5_s50 \
  --task-suite libero_goal \
  --n-trials-per-task 10 \
  --output-dir /content/bivla_eval_libero
```

Baseline done, run the same command with our test-time interventions added
on top:

```bash
# chunk-exec: execute the first 5 of the 10 predicted actions before
# calling the model again (halves forward-pass count)
!python eval_libero.py \
  --emu-hub /content/UNIVLA_LIBERO_IMG_BS192_8K/UNIVLA_LIBERO_IMG_BS192_8K \
  --vq-hub /content/BiVLA/pretrain/Emu3-VisionTokenizer \
  --fast-path /content/BiVLA/UniVLA/pretrain/fast_bridge_t5_s50 \
  --task-suite libero_goal --n-trials-per-task 10 \
  --exec-chunk 5 --output-dir /content/bivla_eval_libero

# foveation (log-polar, matching the log-polar variant already run on
# Bridge/SimplerEnv for OpenVLA/SpatialVLA/UniVLA)
!python eval_libero.py \
  --emu-hub /content/UNIVLA_LIBERO_IMG_BS192_8K/UNIVLA_LIBERO_IMG_BS192_8K \
  --vq-hub /content/BiVLA/pretrain/Emu3-VisionTokenizer \
  --fast-path /content/BiVLA/UniVLA/pretrain/fast_bridge_t5_s50 \
  --task-suite libero_goal --n-trials-per-task 10 \
  --foveate --foveate-mode logpolar --foveate-keep-percent 20 \
  --output-dir /content/bivla_eval_libero

# foveation (blur variant)
!python eval_libero.py \
  --emu-hub /content/UNIVLA_LIBERO_IMG_BS192_8K/UNIVLA_LIBERO_IMG_BS192_8K \
  --vq-hub /content/BiVLA/pretrain/Emu3-VisionTokenizer \
  --fast-path /content/BiVLA/UniVLA/pretrain/fast_bridge_t5_s50 \
  --task-suite libero_goal --n-trials-per-task 10 \
  --foveate --foveate-mode blur --foveate-keep-percent 20 \
  --output-dir /content/bivla_eval_libero
```

`--task-suite` accepts `libero_spatial`, `libero_object`, `libero_goal`,
`libero_10`, `libero_90` (10 tasks each except libero_90). Each run writes a
`summary_<suite>_<timestamp>.json` to `--output-dir` with per-episode results
and the overall success rate.

## Known open question

Whether the local `fast_bridge_t5_s50` FAST tokenizer (reused as-is for
LIBERO in step 5) actually matches the vocabulary the LIBERO checkpoint was
trained against — see the caveat in step 5. Not verified end-to-end since
this sandbox can't load the actual checkpoint.

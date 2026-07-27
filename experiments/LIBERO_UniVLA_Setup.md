# UniVLA (baaivision / Emu3) on LIBERO — Colab setup

Run this in Colab or any machine with real internet access and a GPU. The
dev sandbox this was written in has huggingface.co blocked by network
policy, so every HF-gated step here was authored there but exercised by you
in Colab; the non-HF parts (LIBERO package behaviour, mujoco signatures,
transformers compatibility of the vendored Emu3 code) were verified
directly in the sandbox.

## Which UniVLA is this?

Two unrelated projects share the name. This setup targets **baaivision/UniVLA**
("Unified Vision-Language-Action Model", ICLR 2026) — the Emu3-based one,
whose code is vendored in this repo (`Emu3MoE`, `Emu3Tokenizer`, cluster
paths under `/share/project/yuqi.wang/...`) and whose checkpoints live under
the HF namespace `Yuqi1997/UniVLA`. It is the same backbone already used for
this project's SimplerEnv/Bridge results, so the LIBERO numbers stay
comparable.

It is **not** OpenDriveLab/UniVLA ("Learning to Act Anywhere with
Task-centric Latent Actions", RSS 2025), which is Prismatic-VLM based with a
VQ-VAE latent action model and `qwbu/*` checkpoints. Those checkpoints will
not load in this code at all. Rule of thumb: `Emu3` ⇒ ours, `prismatic` /
`latent action` / `qwbu` ⇒ the other project.

## Version pins matter more than anything else here

The vendored Emu3 modeling code (`UniVLA/reference/Emu3/emu3/mllm/`) is
LLaMA-style modeling code written against **transformers ~4.38–4.44**. On
transformers 5.x it breaks in a long chain of separate places — RoPE config
keys, `GenerationMixin` no longer inherited by `PreTrainedModel`,
`Cache.seen_tokens` / `get_usable_length` removals,
`DynamicCache.from_legacy_cache` removal, and `generate()` now injecting
full-length `position_ids` every step. Those are all patched in this repo
(so 5.x gets *further*), but pinning is the fix that avoids the whole class
of problem.

Verified in the sandbox: on `transformers==4.44.2` the vendored code imports
with **no compatibility shims at all**, and cached `generate()` reproduces
uncached greedy decoding token-for-token in both fp32 and bf16.

`transformers==4.44.2` requires `huggingface_hub<1.0` and `tokenizers<0.20`,
so pin all three together in one pip call or pip will fight itself. Verified
working set: transformers 4.44.2 / tokenizers 0.19.1 / huggingface_hub
0.36.2 / numpy 2.5.1.

---

## 1. Clone BiVLA

`eval_libero.py` and `inference_libero.py` live on the
`claude/serene-davinci-sy33re` branch, not `main`.

```bash
%cd /content
!git clone -b claude/serene-davinci-sy33re https://github.com/trillion-boy/bivla.git BiVLA
%cd /content/BiVLA
!git log -1 --oneline
```

If you already have `/content/BiVLA` from an earlier session:

```bash
%cd /content/BiVLA
!git fetch origin claude/serene-davinci-sy33re
!git checkout claude/serene-davinci-sy33re
!git pull
```

If the repo is private and the clone fails on auth, use whatever method you
already used for the SimplerEnv runs (PAT in the URL, SSH deploy key), then
`git checkout claude/serene-davinci-sy33re`.

## 2. System packages (headless MuJoCo rendering)

```bash
!apt-get -qq update && apt-get -qq install -y libgl1-mesa-dri libegl1 libglvnd0 libgles2
```

## 3. Python packages, with the pins that matter

`pip install libero` only requires `mujoco>=3.0.0`, which resolves to 3.10.x
— but `robosuite==1.4.0` (pinned by `libero`) calls `mujoco.mj_fullM()` with
the pre-3.10 argument order `(model, dst, M)`. Mujoco 3.10.0 changed that to
`(model, data, dst)`, so any real env step or reset dies with `TypeError:
mj_fullM(): incompatible function arguments`. Verified by testing the
signature across versions directly: unchanged through 3.9.0, broken from
3.10.0 on.

```bash
!pip install -q libero
!pip install -q "mujoco==3.9.0"
!pip install -q "transformers==4.44.2" "tokenizers<0.20" "huggingface_hub<1.0"
```

## 4. Restart the runtime — required

Colab preloads `transformers`, and step 3 likely swapped `mujoco` under a
already-imported module. Restart before running anything else:

**Runtime → Restart session**, then continue from step 5. Do not re-run
steps 1–3.

## 5. Pre-seed the LIBERO config

LIBERO prompts interactively on first import, which raises `EOFError` in a
notebook. Write its config file first, using plain Python with no `libero`
import:

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
print("libero config written")
```

## 6. Download the 3D assets

The `libero` pip package ships task definitions (bddl files) but not the 3D
scene/object assets; those pull from HF the first time an env is built.

```python
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

bm = benchmark.get_benchmark_dict()["libero_goal"]()
task = bm.get_task(0)
bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=128, camera_widths=128)
env.reset()
print("LIBERO env OK — assets downloaded")
env.close()
```

## 7. Download the three model artifacts

They are three *separate* things and mixing them up produces confusing
errors. One cell does all three and verifies each:

```python
import os
from huggingface_hub import snapshot_download

# 1. The LIBERO-finetuned Emu3MoE policy (~17GB).
#    snapshot_download mirrors the repo's internal folder layout under
#    local_dir, so the files land in a NESTED subfolder of the same name.
snapshot_download(
    repo_id="Yuqi1997/UniVLA",
    allow_patterns=["UNIVLA_LIBERO_IMG_BS192_8K/*"],
    local_dir="/content/UNIVLA_LIBERO_IMG_BS192_8K",
)
EMU = "/content/UNIVLA_LIBERO_IMG_BS192_8K/UNIVLA_LIBERO_IMG_BS192_8K"

# 2. The frozen Emu3 vision tokenizer. General-purpose, shared across every
#    UniVLA checkpoint — it is NOT inside the LIBERO checkpoint folder.
VISION = "/content/BiVLA/pretrain/Emu3-VisionTokenizer"
snapshot_download(repo_id="BAAI/Emu3-VisionTokenizer", local_dir=VISION)

# 3. The FAST+ action tokenizer. The authors' own LIBERO inference example
#    (UniVLA/models/inference/inference_action.py:48) uses the stock
#    universal FAST+ tokenizer, NOT the Bridge-refit fast_bridge_t5_s50
#    that sits in this repo (that one has a different BPE vocabulary:
#    1024 vs 2048).
FAST = "/content/BiVLA/UniVLA/pretrain/fast"
snapshot_download(repo_id="physical-intelligence/fast", local_dir=FAST)

for name, path, marker in [
    ("emu-hub",   EMU,    "config.json"),
    ("vq-hub",    VISION, "preprocessor_config.json"),
    ("fast-path", FAST,   "processor_config.json"),
]:
    ok = os.path.exists(os.path.join(path, marker))
    print(("OK  " if ok else "FAIL") + f"  {name}: {path}")
    assert ok, f"{name} missing {marker} — check the listing above"
print("all three artifacts present")
```

`eval_libero.py` re-checks these three paths itself before loading the
model, so a path mistake fails in seconds with a clear message instead of
after a multi-minute load.

## 8. Smoke test — one task, one trial

Always do this before committing to a long run.

```bash
%cd /content/BiVLA/adaptive_sparse_vla
!python eval_libero.py \
  --emu-hub /content/UNIVLA_LIBERO_IMG_BS192_8K/UNIVLA_LIBERO_IMG_BS192_8K \
  --vq-hub /content/BiVLA/pretrain/Emu3-VisionTokenizer \
  --fast-path /content/BiVLA/UniVLA/pretrain/fast \
  --task-suite libero_goal --task-ids 0 --n-trials-per-task 1 \
  --save-video --output-dir /content/bivla_eval_libero
```

What a healthy run prints:

- `[action-window] fast_vocab=2048 ids=[...] tok_pad=151643 cfg_pad=151643`
- `[heartbeat] call N ...` every 5 model calls, so you can see it is alive
- `[debug] gen_len=<under 50> eoa=True ...` on the first call

`eoa=True` is the key health signal: the policy emitted its end-of-action
token, meaning the FAST vocabulary lines up and the decoded chunk is real.
If you instead see `gen_len=50 eoa=False` with all-identical `raw_ids`, stop
and go to the diagnostics section below.

On an L4, expect roughly 6 s per model call and ~4–5 min per 300-step
episode.

Watch the rollout:

```python
from IPython.display import Image as IPImage
import glob
IPImage(sorted(glob.glob("/content/bivla_eval_libero/*.gif"))[-1])
```

## 9. The four experiment runs

Only start these once the smoke test looks healthy. At ~6 s/call, a full
10-task × 10-trial suite is on the order of 6–8 hours per variant, so
consider `--n-trials-per-task 5` or a task subset for a first pass.

```bash
%cd /content/BiVLA/adaptive_sparse_vla
EMU=/content/UNIVLA_LIBERO_IMG_BS192_8K/UNIVLA_LIBERO_IMG_BS192_8K
VQ=/content/BiVLA/pretrain/Emu3-VisionTokenizer
FAST=/content/BiVLA/UniVLA/pretrain/fast
OUT=/content/bivla_eval_libero

# baseline
!python eval_libero.py --emu-hub $EMU --vq-hub $VQ --fast-path $FAST \
  --task-suite libero_goal --n-trials-per-task 10 --output-dir $OUT

# chunk-exec: execute only the first 5 of the 10 predicted actions before
# re-querying the model (halves forward passes)
!python eval_libero.py --emu-hub $EMU --vq-hub $VQ --fast-path $FAST \
  --task-suite libero_goal --n-trials-per-task 10 \
  --exec-chunk 5 --output-dir $OUT

# foveation, log-polar
!python eval_libero.py --emu-hub $EMU --vq-hub $VQ --fast-path $FAST \
  --task-suite libero_goal --n-trials-per-task 10 \
  --foveate --foveate-mode logpolar --foveate-keep-percent 20 --output-dir $OUT

# foveation, blur
!python eval_libero.py --emu-hub $EMU --vq-hub $VQ --fast-path $FAST \
  --task-suite libero_goal --n-trials-per-task 10 \
  --foveate --foveate-mode blur --foveate-keep-percent 20 --output-dir $OUT
```

`--task-suite` accepts `libero_spatial`, `libero_object`, `libero_goal`,
`libero_10`, `libero_90`. Each run writes
`summary_<suite>_<timestamp>.json` to `--output-dir` with per-episode
results, mean inference latency, and the overall success rate.

---

## Diagnostics

Set `BIVLA_PROBE=1` to make the policy dump its internal state on every
call — a parameter-wide NaN scan and per-layer hidden-state check at load
time, then prompt token ids, logits health, and an unconstrained generation
sample per step:

```bash
!BIVLA_PROBE=1 python eval_libero.py --emu-hub $EMU --vq-hub $VQ --fast-path $FAST \
  --task-suite libero_goal --task-ids 0 --n-trials-per-task 1 --output-dir $OUT
```

Reading the output:

| Symptom | Meaning |
|---|---|
| `param scan: nan_total>0` | Weights failed to load — checkpoint or transformers-version problem, not a policy problem |
| `param scan` clean but `first NaN hidden layer` set | Numerical break inside the transformer stack at that layer |
| hidden states clean but `logits: nan=<vocab size>` | `lm_head` never got real weights (weight-tying / loading issue) |
| `unconstrained ids: [0, 0, 0, ...]` | Degenerate logits (all-NaN or all-equal), *not* a tokenizer problem |
| `gen_len=50 eoa=False`, identical `raw_ids` | Action-token window or FAST vocabulary mismatch |

The all-NaN-logits failure mode was observed on transformers 5.x and is the
reason for the 4.44.2 pin in step 3.

## Known failure modes and their fixes

| Error | Cause | Fix |
|---|---|---|
| `mj_fullM(): incompatible function arguments` | mujoco ≥3.10 vs robosuite 1.4.0 | `mujoco==3.9.0` (step 3) |
| `EOFError` on first `import libero.libero` | interactive first-run prompt | pre-seed config (step 5) |
| `eglQueryString` AttributeError | missing EGL libs | step 2 packages |
| `Can't load image processor ... no preprocessor_config.json` | `--vq-hub` pointed at the LIBERO checkpoint | point it at Emu3-VisionTokenizer |
| `Repo id must be in the form ...` | a local path that does not exist; transformers falls back to treating it as a HF repo id | check the path actually exists |
| `no file named model.safetensors` | `snapshot_download` nesting | use the nested `.../UNIVLA_LIBERO_IMG_BS192_8K/UNIVLA_LIBERO_IMG_BS192_8K` path |
| `Input type (float) and bias type (c10::BFloat16)` | fixed in `inference_libero.py` | `git pull` |
| `KeyError: 'type'` / `'factor'`, `no attribute 'generate'`, `'seen_tokens'`, `Key and Value must have the same sequence length` | transformers 5.x API drift | patched in-repo, and avoided entirely by the 4.44.2 pin |

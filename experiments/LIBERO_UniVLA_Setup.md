# UniVLA (baaivision / Emu3) on LIBERO — Colab setup

Copy-paste cells for a **completely fresh runtime**. Run this in its own
Colab session: UniVLA pins `transformers==4.44.2` while OpenVLA needs
`4.40.1`, so the two backbones must not share a runtime.

## Which UniVLA

This targets **baaivision/UniVLA** ("Unified Vision-Language-Action Model",
ICLR 2026) — the Emu3-based one whose code is vendored in this repo
(`Emu3MoE`, `Emu3Tokenizer`) and whose checkpoints live under
`Yuqi1997/UniVLA`. Same backbone as this project's SimplerEnv/Bridge
results, so the LIBERO numbers stay comparable.

It is **not** OpenDriveLab/UniVLA (Prismatic-based, `qwbu/*` checkpoints).
Rule of thumb: `Emu3` ⇒ ours, `prismatic` / `latent action` / `qwbu` ⇒ the
other project.

## Why UniVLA matters for this paper

OpenVLA predicts one action per forward pass; UniVLA predicts a **10-step
chunk**. That is the whole point of running both: the efficiency
intervention takes a different form on each architecture, which is exactly
the architecture-dependence claim.

| | OpenVLA | UniVLA |
|---|---|---|
| action chunking | none (1 step) | native, 10 steps |
| compute reduction | `--action-repeat 2` | `--exec-chunk 5` (real chunk-exec) |
| foveation | identical | identical |

OpenVLA reference results on `libero_spatial` (this harness, 5 initial
states/task): baseline 74%, action-repeat 2 → 66%, blur-20% → 58%,
logpolar-20% → 0%.

## What was hard-won (do not re-litigate these)

- **transformers must be 4.44.2.** The vendored Emu3 code is LLaMA-style
  modeling written against ~4.38–4.44. On transformers 5.x it emits
  **all-NaN logits** and the policy outputs token 0 forever. Verified in a
  sandbox: on 4.44.2 the vendored code imports with no shims and cached
  `generate()` matches uncached greedy decoding token-for-token.
- **Rendering must be OSMesa.** EGL hangs or segfaults once a policy is
  resident on the GPU (observed identically with UniVLA and OpenVLA), and
  robosuite 1.4.0 maps `glx` onto GLFW so that is no escape either.
- **Three separate hubs.** `emu_hub`, `vq_hub` (text tokenizer, from
  `BAAI/Emu3-Stage1`) and `vision_hub` (`BAAI/Emu3-VisionTokenizer`) are
  different checkpoints. Collapsing them produces either
  "no preprocessor_config.json" or a silently wrong tokenizer.
- **FAST tokenizer is the stock universal one** (`physical-intelligence/fast`),
  not this repo's Bridge-refit `fast_bridge_t5_s50` (vocab 1024 vs 2048).

---

## 1. Clone BiVLA

```bash
%cd /content
!git clone -b claude/serene-davinci-sy33re https://github.com/trillion-boy/bivla.git BiVLA
%cd /content/BiVLA
!git log -1 --oneline
```

## 2. System packages (OSMesa CPU rendering)

```bash
!apt-get -qq update && apt-get -qq install -y \
    libgl1-mesa-dri libegl1 libglvnd0 libgles2 libosmesa6 libosmesa6-dev
```

## 3. Python packages

`libero` pins `robosuite==1.4.0`, which calls `mujoco.mj_fullM()` with the
pre-3.10 argument order; mujoco 3.10 changed that signature, so any env step
dies with `TypeError: mj_fullM(): incompatible function arguments`.
`tiktoken` is required by `Emu3Tokenizer`.

```bash
!pip install -q libero
!pip install -q "mujoco==3.9.0"
!pip install -q "transformers==4.44.2" "tokenizers<0.20" "huggingface_hub<1.0" \
               "tiktoken==0.6.0" accelerate
!pip install -q PyOpenGL PyOpenGL_accelerate
```

## 4. Restart the runtime — required

Step 3 swapped `mujoco` and `transformers` under already-imported modules.
**Runtime → Restart session**, then continue at step 5. Do not re-run 1–3.

## 5. LIBERO config + 3D assets

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
os.environ["MUJOCO_GL"] = "osmesa"
os.environ["PYOPENGL_PLATFORM"] = "osmesa"

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

## 6. Download the four artifacts

```python
import os
from huggingface_hub import snapshot_download

# 1. LIBERO-finetuned Emu3MoE policy (~17GB).
#    snapshot_download mirrors the repo's folder layout under local_dir, so
#    the files land in a NESTED subfolder of the same name.
snapshot_download(
    repo_id="Yuqi1997/UniVLA",
    allow_patterns=["UNIVLA_LIBERO_IMG_BS192_8K/*"],
    local_dir="/content/UNIVLA_LIBERO_IMG_BS192_8K",
)
EMU = "/content/UNIVLA_LIBERO_IMG_BS192_8K/UNIVLA_LIBERO_IMG_BS192_8K"

# 2. Base Emu3 — only the TEXT TOKENIZER is needed, so skip the weights.
#    Emu3-Stage1 is an 8B checkpoint; downloading it whole wastes ~16GB.
VQ = "/content/pretrain/Emu3-Stage1"
snapshot_download(repo_id="BAAI/Emu3-Stage1", local_dir=VQ,
                  ignore_patterns=["*.safetensors", "*.bin", "*.pt", "*.pth"])

# 3. Frozen Emu3 vision tokenizer (~1GB).
VISION = "/content/pretrain/Emu3-VisionTokenizer"
snapshot_download(repo_id="BAAI/Emu3-VisionTokenizer", local_dir=VISION)

# 4. Stock universal FAST+ action tokenizer (small).
FAST = "/content/BiVLA/UniVLA/pretrain/fast"
snapshot_download(repo_id="physical-intelligence/fast", local_dir=FAST)

for name, path, marker in [
    ("emu-hub",    EMU,    "config.json"),
    ("vq-hub",     VQ,     "tokenizer_config.json"),
    ("vision-hub", VISION, "preprocessor_config.json"),
    ("fast-path",  FAST,   "processor_config.json"),
]:
    ok = os.path.exists(os.path.join(path, marker))
    print(("OK  " if ok else "FAIL") + f"  {name}: {path}")
    if not ok:
        print("      contents:", sorted(os.listdir(path))[:15])
print("done")
```

If `vq-hub` reports FAIL, the tokenizer file set differs from what the
`ignore_patterns` download kept — list the repo and adjust:

```python
from huggingface_hub import list_repo_files
print([f for f in list_repo_files("BAAI/Emu3-Stage1") if "safetensors" not in f])
```

## 7. Smoke test — one task, one trial

```bash
%cd /content/BiVLA/adaptive_sparse_vla
!python eval_libero.py --backbone univla --mujoco-gl osmesa \
  --emu-hub /content/UNIVLA_LIBERO_IMG_BS192_8K/UNIVLA_LIBERO_IMG_BS192_8K \
  --vq-hub /content/pretrain/Emu3-Stage1 \
  --vision-hub /content/pretrain/Emu3-VisionTokenizer \
  --fast-path /content/BiVLA/UniVLA/pretrain/fast \
  --task-suite libero_spatial --task-ids 0 --n-trials-per-task 1 \
  --save-video --output-dir /content/bivla_eval_libero_univla
```

**The one thing to check: `eoa=True` in the `[debug]` line.** It means the
policy emitted its end-of-action token, i.e. the FAST vocabulary lines up
and the decoded chunk is real. This was never verifiable before, because
the all-NaN logits under transformers 5.x masked it.

| `[debug]` shows | meaning |
|---|---|
| `eoa=True`, `chunk_shape=(10, 7)` | decoding is correct, proceed |
| `gen_len=50 eoa=False`, identical `raw_ids` | FAST vocabulary or token-window mismatch |

Also expect `[action-window] ... tok_pad=151643 cfg_pad=151643` at load,
and `predict_action_frames=10`.

## 8. The four experiment runs

Match OpenVLA's protocol exactly — `libero_spatial`, 5 initial states per
task, 50 episodes per condition — so the two backbones are comparable.

```bash
%cd /content/BiVLA/adaptive_sparse_vla
EMU=/content/UNIVLA_LIBERO_IMG_BS192_8K/UNIVLA_LIBERO_IMG_BS192_8K
VQ=/content/pretrain/Emu3-Stage1
VIS=/content/pretrain/Emu3-VisionTokenizer
FAST=/content/BiVLA/UniVLA/pretrain/fast
OUT=/content/bivla_eval_libero_univla
COMMON="--backbone univla --mujoco-gl osmesa --emu-hub $EMU --vq-hub $VQ \
        --vision-hub $VIS --fast-path $FAST --task-suite libero_spatial \
        --n-trials-per-task 5 --output-dir $OUT"

# 1) baseline — executes all 10 predicted actions
!python eval_libero.py $COMMON

# 2) chunk-exec — executes the first 5 of 10, halving forward passes.
#    This is the real chunk-exec that OpenVLA cannot do.
!python eval_libero.py $COMMON --exec-chunk 5

# 3) foveation, log-polar
!python eval_libero.py $COMMON --foveate --foveate-mode logpolar --foveate-keep-percent 20

# 4) foveation, blur
!python eval_libero.py $COMMON --foveate --foveate-mode blur --foveate-keep-percent 20
```

Each run writes `summary_libero_spatial_<timestamp>.json` with `backbone`,
`checkpoint`, `exec_chunk`, `action_repeat`, foveation settings, per-episode
results, mean latency and success rate.

Budget: OpenVLA averaged ~520 ms/inference and ~2 min/episode on an A100
with OSMesa. UniVLA is a larger model with two camera views, so expect
slower; run the smoke test first and extrapolate from its ms/infer before
committing to all four conditions.

## 9. Compare the two backbones

Upload the summary JSONs from both backbones into one cell:

```python
import json
from google.colab import files

uploaded = files.upload()
rows = []
for name, content in uploaded.items():
    d = json.loads(content.decode("utf-8"))
    fov = d["foveate"]
    if fov["enabled"]:
        cond = f"foveate-{fov['mode']} {fov['keep_percent']:.0f}%"
    elif d.get("exec_chunk", 0) > 0:
        cond = f"chunk-exec {d['exec_chunk']}/{d['predict_action_frames']}"
    elif d["action_repeat"] > 1:
        cond = f"action-repeat {d['action_repeat']}"
    else:
        cond = "baseline"
    per_task = {}
    for e in d["episodes"]:
        per_task.setdefault(e["task_id"], [0, 0])
        per_task[e["task_id"]][1] += 1
        per_task[e["task_id"]][0] += int(e["success"])
    rows.append({
        "backbone": d["backbone"], "condition": cond,
        "sr": d["success_rate"] * 100, "n": d["n_episodes"],
        "ms": round(d["avg_model_ms_per_infer"]),
        "per_task": {k: f"{v[0]}/{v[1]}" for k, v in per_task.items()},
    })

rows.sort(key=lambda r: (r["backbone"], r["condition"]))
print(f"{'backbone':10s} {'condition':24s} {'SR%':>6s} {'n':>4s} {'ms':>5s}")
for r in rows:
    print(f"{r['backbone']:10s} {r['condition']:24s} {r['sr']:6.1f} {r['n']:4d} {r['ms']:5d}")
```

## Troubleshooting

| symptom | cause | fix |
|---|---|---|
| all-NaN logits, generated ids all `0` | transformers 5.x | pin `4.44.2` (step 3) |
| silent death / hang at `building env` | EGL vs CUDA | `--mujoco-gl osmesa` |
| `Can't load image processor ... no preprocessor_config.json` | `--vision-hub` wrong | point at Emu3-VisionTokenizer |
| `Could not load tokenizer from subfolder bpe_tokenizer` | tokenizer read from the LIBERO checkpoint | `--vq-hub` = Emu3-Stage1 |
| `gen_len=50 eoa=False`, identical ids | wrong FAST tokenizer | use `physical-intelligence/fast` |
| `no file named model.safetensors` | snapshot_download nesting | use the doubled `.../UNIVLA_LIBERO_IMG_BS192_8K/UNIVLA_LIBERO_IMG_BS192_8K` path |
| `mj_fullM(): incompatible function arguments` | mujoco ≥ 3.10 | `mujoco==3.9.0` |

`eval_libero.py` also repairs a truncated `~/.libero/config.yaml`, validates
every checkpoint path before loading the model, ends episodes as soon as
LIBERO reports success, and dumps thread stacks if no model call happens for
5 minutes.

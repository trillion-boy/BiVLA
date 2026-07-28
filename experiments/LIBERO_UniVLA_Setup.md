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

The shared axis is **env steps executed per forward pass** -- higher means
more open-loop execution and less compute. The two architectures sit at
opposite ends of it by default, and the lever each one offers points in a
different direction:

| | OpenVLA | UniVLA |
|---|---|---|
| action chunking | none (1 step) | native, 10 steps |
| baseline: env steps / forward | 1 (fully closed-loop) | 10 (already amortized) |
| cheaper lever | `--action-repeat 2` → 2 steps/forward | `--action-repeat 2` → **20** steps/forward |
| more-reactive lever | none (already 1 step) | `--exec-chunk 5` → 5 steps/forward, 2x the compute |
| foveation | identical | identical |

That asymmetry is not a flaw in the comparison -- it *is* the
architecture-dependence result. A non-chunking policy can only be made
cheaper; a chunked policy is already cheap and can only be made more
reactive.

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

## 6.5. Patch the FAST tokenizer — required

`physical-intelligence/fast` is the stock FAST+ release. The UniVLA authors
added a robustness fix to *their* copy that the stock release does not have,
and without it ~4.5% of action chunks are silently corrupted.

FAST encodes an action chunk as DCT coefficients -> characters -> BPE tokens.
Nothing structurally forces the model's generated BPE sequence to decode back
to exactly `time_horizon * action_dim` characters, so it occasionally lands
one or two short or long. The stock `decode()` then fails the reshape, and
its except-block substitutes all-zero DCT coefficients -- which after
un-normalization is **not** a no-op but a fixed drift
(`[0.116, 0.033, 0, 0.009, 0.014, 0.056, -1]`), so the arm keeps moving for a
full 10-step chunk on a dead command. Measured at 4.4-4.7%; patched, 0%.

Insert only the length fix. Do **not** copy the whole file from
`fast_bridge_t5_s50`: that copy also carries different quantization defaults
(`scale=50, min_token=-112` vs the stock `scale=10, min_token=0`), and the
stock values are the ones this checkpoint decodes correctly with.

```python
p = "/content/BiVLA/UniVLA/pretrain/fast/processing_action_tokenizer.py"
src = open(p).read()

if "max_length = self.time_horizon * self.action_dim" in src:
    print("already patched")
else:
    open(p + ".bak", "w").write(src)
    old = "                decoded_dct_coeff = decoded_dct_coeff.reshape(-1, self.action_dim)"
    new = """                max_length = self.time_horizon * self.action_dim
                if len(decoded_dct_coeff) > max_length:
                    decoded_dct_coeff = decoded_dct_coeff[:max_length]
                elif len(decoded_dct_coeff) < max_length:
                    decoded_dct_coeff = np.pad(
                        decoded_dct_coeff, (0, max_length - len(decoded_dct_coeff)),
                        mode="constant")
                decoded_dct_coeff = decoded_dct_coeff.reshape(-1, self.action_dim)"""
    assert src.count(old) == 1, f"insertion point found {src.count(old)} times"
    open(p, "w").write(src.replace(old, new))
    print("patched (backup at .bak)")

import subprocess
print(subprocess.run(["grep", "-n", "scale: float|min_token: int", "-E", p],
                     capture_output=True, text=True).stdout)
```

The last line must still print `scale: float = 10` and `min_token: int = 0`.
Confirm with the smoke test below: `[decode] FAST failures: 0/N (0.0%)`.

**This lives in `/content` and is lost on every runtime restart.** Re-apply it
after any restart, before running evals.

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
```

Each condition is its own cell. Shell-style variable assignments do not work
in a Colab cell (`EMU=/content/...` is a Python syntax error), so every
command carries the full paths.

```bash
# 1) baseline -- executes all 10 predicted actions
!python eval_libero.py --backbone univla --mujoco-gl osmesa \
  --emu-hub /content/UNIVLA_LIBERO_IMG_BS192_8K/UNIVLA_LIBERO_IMG_BS192_8K \
  --vq-hub /content/pretrain/Emu3-Stage1 \
  --vision-hub /content/pretrain/Emu3-VisionTokenizer \
  --fast-path /content/BiVLA/UniVLA/pretrain/fast \
  --task-suite libero_spatial --n-trials-per-task 5 \
  --output-dir /content/bivla_eval_libero_univla
```

```bash
# 2) action-repeat 2 -- holds each of the 10 predicted actions for 2 env
#    steps, so one forward now covers 20 env steps: 2x cheaper. This is the
#    SAME mechanism (np.repeat) applied to OpenVLA, which is what makes the
#    two backbones comparable under one intervention.
!python eval_libero.py --backbone univla --mujoco-gl osmesa \
  --emu-hub /content/UNIVLA_LIBERO_IMG_BS192_8K/UNIVLA_LIBERO_IMG_BS192_8K \
  --vq-hub /content/pretrain/Emu3-Stage1 \
  --vision-hub /content/pretrain/Emu3-VisionTokenizer \
  --fast-path /content/BiVLA/UniVLA/pretrain/fast \
  --task-suite libero_spatial --n-trials-per-task 5 \
  --action-repeat 2 \
  --output-dir /content/bivla_eval_libero_univla
```

```bash
# 3) foveation, log-polar
!python eval_libero.py --backbone univla --mujoco-gl osmesa \
  --emu-hub /content/UNIVLA_LIBERO_IMG_BS192_8K/UNIVLA_LIBERO_IMG_BS192_8K \
  --vq-hub /content/pretrain/Emu3-Stage1 \
  --vision-hub /content/pretrain/Emu3-VisionTokenizer \
  --fast-path /content/BiVLA/UniVLA/pretrain/fast \
  --task-suite libero_spatial --n-trials-per-task 5 \
  --foveate --foveate-mode logpolar --foveate-keep-percent 20 \
  --output-dir /content/bivla_eval_libero_univla
```

```bash
# 4) foveation, blur
!python eval_libero.py --backbone univla --mujoco-gl osmesa \
  --emu-hub /content/UNIVLA_LIBERO_IMG_BS192_8K/UNIVLA_LIBERO_IMG_BS192_8K \
  --vq-hub /content/pretrain/Emu3-Stage1 \
  --vision-hub /content/pretrain/Emu3-VisionTokenizer \
  --fast-path /content/BiVLA/UniVLA/pretrain/fast \
  --task-suite libero_spatial --n-trials-per-task 5 \
  --foveate --foveate-mode blur --foveate-keep-percent 20 \
  --output-dir /content/bivla_eval_libero_univla
```

Optional 5th condition, unique to a chunked policy -- `--exec-chunk 5`
executes only 5 of the 10 predicted actions before re-querying, doubling
forward passes to double the re-planning rate. It answers whether UniVLA's
default 10-step open-loop execution is costing accuracy. Run it only after
the four above, since OpenVLA has no counterpart to compare it against.

Each run writes `summary_libero_spatial_<timestamp>.json` with `backbone`,
`checkpoint`, `exec_chunk`, `action_repeat`, foveation settings, the `depth`
block, per-episode results, mean latency and success rate.

## 8.5. The depth axis — the only lever that cuts UniVLA's latency

The four runs above cover the temporal axis (`--action-repeat`) and the
spatial axis (`--foveate`). Neither can make UniVLA faster:

- **Temporal is already spent.** UniVLA's baseline executes 10 env steps per
  forward. `--action-repeat 2` takes that to 20 and success collapses to 28%.
- **Spatial does not touch wall-clock.** Foveation degrades pixels but the
  image still becomes the same number of VQ tokens, so ms/inference is flat
  (1882 -> 1888). It reduces information, not compute. Cutting the tokens
  themselves does not help either: `docs/VISUAL_TOKENS_VS_LATENCY.md` profiles
  a UniVLA step as 6% VQ encode / 13% prefill / **70% autoregressive decode**,
  so the entire visual path is a ~19% ceiling — and FastV, measured, destroyed
  success (100->75->38%) while leaving latency at 1.0x.

The decode is 70% of the step, and it pays for **every layer on every token**.
Bypassing redundant layers is the one intervention that shrinks it. Rank each
decoder layer by `1 - cos(layer_in, layer_out)` — low means the layer barely
changes the representation — and replace the most redundant ones with a
pass-through. Training-free, no external module.

This is validated on the same Emu3 backbone in SimplerEnv
(`docs/DEPTH_PRUNING_RESULTS.md`): success **74% -> 78-81%** at **1.10-1.25x**
speedup, i.e. it often *raises* accuracy while cutting latency. It is also
known to be backbone-dependent — the same mechanism on SpatialVLA's Gemma2
hurt 3 of 4 tasks at a single bypassed layer — which is exactly the
architecture-dependence claim, now on a third axis.

Start with static pruning, since it has one knob and the SimplerEnv sweep
says 4 and 8 bracket the useful range:

```bash
# 5) depth-prune 4 -- bypass the 4 most redundant of Emu3's 32 layers
!python eval_libero.py --backbone univla --mujoco-gl osmesa \
  --emu-hub /content/UNIVLA_LIBERO_IMG_BS192_8K/UNIVLA_LIBERO_IMG_BS192_8K \
  --vq-hub /content/pretrain/Emu3-Stage1 \
  --vision-hub /content/pretrain/Emu3-VisionTokenizer \
  --fast-path /content/BiVLA/UniVLA/pretrain/fast \
  --task-suite libero_spatial --n-trials-per-task 5 \
  --depth-prune 4 \
  --output-dir /content/bivla_eval_libero_univla
```

```bash
# 6) depth-prune 8 -- the aggressive end; SimplerEnv got ~1.25x here
!python eval_libero.py --backbone univla --mujoco-gl osmesa \
  --emu-hub /content/UNIVLA_LIBERO_IMG_BS192_8K/UNIVLA_LIBERO_IMG_BS192_8K \
  --vq-hub /content/pretrain/Emu3-Stage1 \
  --vision-hub /content/pretrain/Emu3-VisionTokenizer \
  --fast-path /content/BiVLA/UniVLA/pretrain/fast \
  --task-suite libero_spatial --n-trials-per-task 5 \
  --depth-prune 8 \
  --output-dir /content/bivla_eval_libero_univla
```

```bash
# 7) phase-adaptive controller -- full depth for the precise approach+grasp,
#    then bypass 8 once the policy commits to closing the gripper. Only worth
#    running if depth-prune 8 loses accuracy that depth-prune 4 keeps: that
#    gap is what the controller exists to recover.
!python eval_libero.py --backbone univla --mujoco-gl osmesa \
  --emu-hub /content/UNIVLA_LIBERO_IMG_BS192_8K/UNIVLA_LIBERO_IMG_BS192_8K \
  --vq-hub /content/pretrain/Emu3-Stage1 \
  --vision-hub /content/pretrain/Emu3-VisionTokenizer \
  --fast-path /content/BiVLA/UniVLA/pretrain/fast \
  --task-suite libero_spatial --n-trials-per-task 5 \
  --depth-ctrl --depth-deep 2 --depth-shallow 8 \
  --output-dir /content/bivla_eval_libero_univla
```

### Reading the result

The number that matters is **`avg_model_ms_per_infer`**, not success alone —
this is the first LIBERO condition where it is supposed to move. Compare
against the baseline run's value (~1882 ms):

| outcome | reading |
|---|---|
| ms drops, success holds | the claim: UniVLA has exploitable depth redundancy |
| ms drops, success drops | trade-off — try `--depth-prune 4`, then `--depth-ctrl` |
| ms flat | the bypass is not taking effect; check the `[depth] calibrated:` line |

At startup the run prints `[depth] static pruning ON: ...` and, on the first
step, `[depth] calibrated: bypass=[...]` with the layer indices actually
replaced. The summary JSON carries the same under a `depth` block. If
`bypassed_layers` is empty, nothing was pruned and the run is just a baseline.

Only the back half of the stack is eligible (`--depth-min-layer 0.5`) and
adjacent layers are avoided while cheaper candidates remain — early layers
perform the large foundational transforms every later layer depends on, and
removing them corrupts the representation outright (measured on Gemma2:
bypassing layers 2 and 4 made generation never terminate).

The bookkeeping is unit-tested on CPU against a real, tiny vendored Emu3 —
including that a cached `generate()` with layers bypassed decodes identically
to uncached greedy decoding, which is what breaks if the placeholder KV
entries are wrong:

```bash
cd /content/BiVLA/adaptive_sparse_vla && python test_depth_libero_logic.py
# -> ALL 16 DEPTH-PRUNING CHECKS PASSED
```

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

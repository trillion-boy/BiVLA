# Incident Report: Colab Driver Rollout Breaks All SAPIEN-Based Simulation

**Date of incident:** 2026-07-14
**Status:** PARTIALLY RESOLVED / root cause revised (see "Update 2026-07-15" below) — on Jul 15 a Colab VM with the *identical* driver (580.82.07, Open Kernel Module) completed full SimplerEnv episodes, so the driver version alone does not determine failure. The Jul 14 failures were real and reproducible that day, but their exact cause is now an open question. Practical guidance: run the preflight check (with BOTH env exports) at the start of every session and proceed if it passes.
**Impact:** All SimplerEnv / ManiSkill2 / SAPIEN 2.2 evaluation (OpenVLA RetinaBased *and* SpatialVLA experiments) failed on every Colab GPU VM obtained on Jul 14 (3/3 VMs across L4 and T4 pools). If the driver-580 hosts now cover the fleet, running this class of experiment on Colab may be difficult until something changes on the platform side.

---

## Update 2026-07-15: rendering works again on an identical-driver VM

The next morning, a fresh Colab L4 VM ran the full OpenVLA smoke test
successfully — complete episodes, rendering included — with `nvidia-smi`
reporting the **same driver as the broken Jul 14 VMs**: 580.82.07, CUDA 13.0,
Open Kernel Module, same Apr 30 build. Facts from that session:

- **Working VM (Jul 15, ~04:17 UTC boot):** driver 580.82.07 OKM; eval run
  with `VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json` and
  `LD_LIBRARY_PATH=/usr/lib64-nvidia` exported; episodes completed normally.
- **Failing VM (Jul 15, ~04:13 UTC, a different VM minutes earlier):**
  preflight segfaulted — but that preflight cell was missing the
  `LD_LIBRARY_PATH` export, so it cannot distinguish "broken VM" from
  "incomplete test". A false negative is plausible.
- The userland driver libraries in the runtime image are timestamped
  **Jul 13 13:51 UTC** (`/usr/lib64-nvidia/libGLX_nvidia.so.580.82.07`),
  corroborating that a new image shipped in the Jul 13–14 window.

**Later the same day (Jul 15, ~05:30 UTC), a third VM confirmed genuine
per-VM variance.** A fresh L4 allocated for the SpatialVLA notebook — same
driver 580.82.07 — failed the full, correctly-configured preflight:

- Its runtime image was barer than the morning VM's: no `libvulkan1`
  installed, `/etc/vulkan/icd.d/` present but **empty** (no NVIDIA ICD).
- After installing the loader and writing ICD files manually: the **GLX ICD
  failed instance creation** (`vk_icdGetInstanceProcAddr` not resolvable →
  `ERROR_INCOMPATIBLE_DRIVER`) while the **EGL ICD worked** — `vulkaninfo`
  enumerated `deviceName = NVIDIA L4` normally.
- With that healthy EGL instance, SAPIEN still **segfaulted at
  `take_picture()`**, with no "incompatible driver" warning — i.e., the
  device was found and the crash matches the Jul 14 `Buffer::map()`
  signature exactly.

So within a single day: one 580.82.07 VM renders fine end-to-end (GLX ICD),
another 580.82.07 VM segfaults despite a fully verified Vulkan stack (EGL
ICD, L4 enumerated). **The failure is real, deterministic per-VM, and not
explained by driver version, ICD choice, loader presence, or env vars.**
The differentiating host-side factor remains unidentified. Which Vulkan
userland pieces are present, and which ICD path works, also varies VM to
VM — consistent with an uneven image rollout.

**Practical takeaway:** treat Colab VMs as a lottery right now. Don't write
off (or trust) a session on the driver string alone — run the full preflight
below on every fresh VM and let the render test decide. If it fails, delete
the runtime and re-roll; working VMs demonstrably exist in the same pool.

---

## TL;DR (as written 2026-07-14 — see update above)

Every Colab GPU VM allocated on July 14 (two L4s, one T4) carried **NVIDIA
driver 580.82.07 (CUDA 13.0, Open Kernel Module, build date 2026-04-30)**,
observed directly via `nvidia-smi`. On this driver, **SAPIEN 2.2.2 segfaults
in `svulkan2::core::Buffer::map()`** the moment a camera image is read back
from GPU memory (`take_picture()`), killing every SimplerEnv episode before
the first step. The exact same code, wheels, and setup completed full
24-episode evaluation runs the night before (July 13). Since the crash is
deterministic on first render, the July 13 VM cannot have been on this driver
— i.e., nothing in this repository changed; the host VM underneath did.
(Whether this was a fleet-wide rollout or a gradual host-pool refresh is
unknown; see "Evidence status" below.)

---

## Evidence status: observed vs. inferred vs. unknown

**Directly observed (certain):**
- `nvidia-smi` on all three VMs allocated Jul 14 (L4 ×2, T4 ×1): driver
  580.82.07, CUDA 13.0, Open Kernel Module.
- SAPIEN 2.2.2 segfault at first `take_picture()` on every one of those VMs,
  with the gdb backtrace below. The crash is deterministic — it fires on the
  very first render call, every time.
- The identical workload (same repo state, same wheels, same setup cells)
  completed 12 full 24-episode runs on a Colab L4 the evening of Jul 13.

**Inferred (strong, but indirect):** the Jul 13 VM was on an older driver.
We did **not** record `nvidia-smi` output on Jul 13, so this is a deduction,
not an observation: a crash that reproduces 100% on the first render could
not have coexisted with 12 completed runs, therefore the Jul 13 host did not
exhibit it — consistent with a pre-570 driver.

**Not found / unknown:**
- **No official announcement exists** (checked Jul 14). Colab has historically
  surfaced runtime-stack upgrades as issues on `googlecolab/colabtools`
  (e.g. [#5061](https://github.com/googlecolab/colabtools/issues/5061) for
  CUDA 12.5 / driver 550, [#6053](https://github.com/googlecolab/colabtools/issues/6053)
  for the Jul 9, 2026 PyTorch 2.13 bump). No analogous issue exists for
  driver 580 / CUDA 13 as of Jul 14, and the
  [backend-info](https://github.com/googlecolab/backend-info) repo tracks
  userland packages only — host driver versions have never been published
  there. So the absence of a notice is not unusual: **Colab does not
  announce host-driver changes**, and the claim that "the driver changed
  between Jul 13 and Jul 14" rests on our inference above, not on any
  Google statement.
- Whether this is a synchronized fleet-wide rollout or a gradual host-pool
  refresh (with Jul 14's allocations simply landing on refreshed hosts) is
  unknown. Three-for-three VMs across two GPU types suggests broad coverage,
  but n=3 cannot distinguish the two.

## Timeline

| When (KST) | Event |
|---|---|
| Jul 13, evening | Full sdpa reproduction suite (12 runs × 24 episodes) completes on Colab L4. Results committed (`results_reproduction_sdpa/`, commit `567268d`). Rendering fully functional. Driver version not recorded. |
| Jul 14, ~11:00 | New Colab session for SpatialVLA experiments. Setup succeeds; smoke test **segfaults** immediately after model load, at env creation. `nvidia-smi`: 580.82.07. |
| Jul 14, afternoon | Systematic debugging across two L4 VMs and one T4 VM (see below). Every VM shows driver 580.82.07; every SAPIEN render attempt segfaults identically. |
| Jul 14, evening | Root cause confirmed with gdb; all workaround avenues exhausted. Incident declared. |

## Symptom

```
[svulkan2] [error] Vulkan is incompatible with your driver. ...   (some configs)
Segmentation fault (core dumped)                                   (all configs)
```

Model loading (CUDA path) works. Physics engine creation works. Renderer object
creation works. The crash is precisely at the first attempt to read a rendered
image back from GPU memory.

## Root cause (gdb backtrace)

```
Thread 1 "python" received signal SIGSEGV, Segmentation fault.
#0 svulkan2::core::Buffer::map()
#1 svulkan2::renderer::Renderer::prepareRender(svulkan2::scene::Camera&)
#2 svulkan2::renderer::Renderer::render(...)
#3 sapien::Renderer::SVulkan2Camera::takePicture()
```

`Buffer::map()` maps GPU memory into host address space. The SAPIEN 2.2.2
binary wheel (compiled ~2023) makes assumptions about the driver's Vulkan
memory-type layout that no longer hold on the 580-series driver, dereferences
an invalid mapping, and is killed by the OS. SAPIEN 2.x is end-of-life
upstream, so a fixed 2.x wheel seems unlikely in the short term (though not
impossible if enough users are affected by the 580 transition), and SAPIEN
3.x is API-incompatible with `ManiSkill2_real2sim` as-is.

Importantly, **Vulkan itself is healthy** on these VMs: with
`LD_LIBRARY_PATH=/usr/lib64-nvidia`, `vulkaninfo` enumerates the L4 as a
conformant Vulkan 1.4.312 device. The incompatibility is specific to
SAPIEN 2.2's memory-mapping code path.

## Hypotheses tested and eliminated

| # | Hypothesis | Test | Result |
|---|---|---|---|
| 1 | Wrong Vulkan ICD file | GLX ICD (`/etc/vulkan/icd.d/nvidia_icd.json`), EGL ICD (`configs/nvidia_icd_egl.json`), software lvp ICD | GLX/EGL: segfault. lvp: `ErrorExtensionNotPresent` (llvmpipe lacks required extensions) |
| 2 | Loader can't find driver libs | `LD_LIBRARY_PATH=/usr/lib64-nvidia` (fixed `vulkaninfo`; L4 enumerates) | vulkaninfo fixed; SAPIEN still segfaults |
| 3 | conda environment artifact | Bare `python3.10 -m venv` + `pip install sapien==2.2.2` only | Identical segfault |
| 4 | Stale/bundled Vulkan loader | `LD_PRELOAD` of system `libvulkan.so.1`; no bundled loader found in envs | Identical segfault |
| 5 | Host hardware (BAR) difference | `nvidia-smi -q` BAR1 comparison across VMs | Identical (32768 MiB) on working-era and broken VMs |
| 6 | Missing Xvfb / DISPLAY | Xvfb started in-cell | Irrelevant (offscreen rendering doesn't need X); segfault persists |
| 7 | GPU-pool specific (L4 only) | Switched runtime to T4 | T4 pool also on 580.82.07 |
| 8 | Colab runtime version | Pinned runtime `2026.04` and `2026.01` | **Driver unchanged (580)** — the runtime selector swaps the userland container only; the GPU driver lives in the host VM layer and has no user-facing control |
| 9 | OOM / RAM | `dmesg`, `free -h` | 51 GiB free; no OOM kill records |

## Secondary (unrelated) issue found the same day

HuggingFace large-file downloads intermittently failed (stalls / HTTP 403).
Cause: `huggingface.co` load-balances LFS downloads between two CDN edges;
the **GCP edge (`us.gcp.cdn.hf.co`) returned 403 from Colab** while the AWS
edge (`cas-bridge.xethub.hf.co`) served at 244 MB/s. Workaround: retry loop
around `snapshot_download(..., max_workers=1)` with
`huggingface_hub.constants.HF_HUB_DOWNLOAD_TIMEOUT = 15` — each retry re-rolls
the CDN edge. This is independent of the driver incident.

## External corroboration

The SAPIEN-vs-driver-≥570 failure mode predates Colab's rollout (reported from
Docker/self-managed machines whose owners upgraded drivers earlier):

- SAPIEN #271 — segfault on A100, driver ≥ 570, ICD overrides ineffective:
  https://github.com/haosulab/SAPIEN/issues/271
- RoboTwin #259 — no rendering device on driver 580.126 + CUDA 13; closed unresolved:
  https://github.com/RoboTwin-Platform/RoboTwin/issues/259
- ManiSkill #1020 — sapien 3.0.0b1 segfault on driver 570.133:
  https://github.com/haosulab/ManiSkill/issues/1020

## Current options

1. **Wait and re-check periodically.** This may resolve itself: Colab could
   adjust or roll back the image, older hosts may still exist in some pools,
   or the SAPIEN/ManiSkill community may ship a workaround now that 580-era
   drivers are spreading. The preflight cells below make each re-check cost
   only a few seconds, so it is cheap to keep trying while working on other
   things. The uncertainty is the timeline — it could be days or much longer.
2. **Rented GPU with a pinned driver image** (RunPod / Lambda / Vast):
   choose a CUDA 12.1 / driver 535–550 template and the problem does not
   exist. Estimated cost for the full remaining experiment plan: **$5–10**.
3. **Lab machine** with driver < 570 (the mentor's original local
   `run_experiment.sh` path works as-is).

## Preflight procedure for any future Colab session

Run **before** any setup. Only the render check is authoritative: as the
Jul 15 update shows, the driver string alone is NOT a reliable predictor
(580.82.07 VMs both failed and worked), and a preflight without the
`LD_LIBRARY_PATH` export can produce a false negative.

Single self-contained cell (~3–4 min on a fresh VM). Hard-learned details
baked in: fresh VMs may lack `python3.10-venv` (and `ensurepip` may be
broken — hence the get-pip bootstrap), may lack `libvulkan1` entirely, may
have an **empty** `/etc/vulkan/icd.d/`, and which NVIDIA ICD works (GLX vs
EGL) varies per VM — so both are written and tried in turn. On success it
prints which ICD to export in all subsequent run cells.

```bash
%%bash
nvidia-smi | grep "Driver Version"
apt-get update -qq
apt-get install -y -qq python3.10-venv libvulkan1 vulkan-tools > /dev/null
if [ ! -f /content/vtest/bin/python ]; then
  python3.10 -m venv /content/vtest --without-pip
  curl -sS https://bootstrap.pypa.io/get-pip.py -o /content/get-pip.py
  /content/vtest/bin/python /content/get-pip.py -q
  /content/vtest/bin/pip install -q "setuptools<81" "numpy<2" sapien==2.2.2
fi
mkdir -p /etc/vulkan/icd.d
cat > /etc/vulkan/icd.d/nvidia_icd.json <<'JSON'
{"file_format_version":"1.0.0","ICD":{"library_path":"libGLX_nvidia.so.0","api_version":"1.3.277"}}
JSON
cat > /etc/vulkan/icd.d/nvidia_egl.json <<'JSON'
{"file_format_version":"1.0.0","ICD":{"library_path":"libEGL_nvidia.so.0","api_version":"1.3.277"}}
JSON
export LD_LIBRARY_PATH=/usr/lib64-nvidia
for icd in nvidia_icd nvidia_egl; do
  export VK_ICD_FILENAMES=/etc/vulkan/icd.d/$icd.json
  echo "== [$icd] render test =="
  /content/vtest/bin/python - << 'EOF'
import sapien.core as sapien
e = sapien.Engine(); r = sapien.SapienRenderer(offscreen_only=True); e.set_renderer(r)
s = e.create_scene(); cam = s.add_camera('c', 128, 128, 1.0, 0.01, 10)
s.step(); s.update_render(); cam.take_picture()
print('RENDER OK')
EOF
  if [ $? -eq 0 ]; then echo ">>> VM USABLE (use VK_ICD_FILENAMES=/etc/vulkan/icd.d/$icd.json in every run cell) <<<"; exit 0; fi
done
echo ">>> VM FAILED preflight -- delete runtime and re-roll <<<"
exit 1
```

## What is NOT affected

- All committed reproduction results and reports (`results_reproduction_*`,
  `REPRODUCTION_REPORT.md`) — complete before the incident.
- The SpatialVLA foveation port (`SpatialVLA/experiments/tome/foveation.py`,
  `--foveate` flag) — code is committed and unit-tested; only *execution* of
  the simulation experiments is blocked.
- Model inference itself (CUDA path) — only Vulkan *rendering* is broken.

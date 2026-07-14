# Incident Report: Colab Driver Rollout Breaks All SAPIEN-Based Simulation

**Date of incident:** 2026-07-14
**Status:** UNRESOLVED — no software-side fix exists; requires an environment with NVIDIA driver < 570
**Impact:** All SimplerEnv / ManiSkill2 / SAPIEN 2.2 evaluation (OpenVLA RetinaBased *and* SpatialVLA experiments) is currently impossible on Google Colab.

---

## TL;DR

Between the night of July 13 and the morning of July 14, Google Colab rolled out
new GPU VM images carrying **NVIDIA driver 580.82.07 (CUDA 13.0, Open Kernel
Module, build date 2026-04-30)**. On this driver, **SAPIEN 2.2.2 segfaults in
`svulkan2::core::Buffer::map()`** the moment a camera image is read back from
GPU memory (`take_picture()`), killing every SimplerEnv episode before the
first step. The exact same code, wheels, and setup completed full 24-episode
evaluation runs the night before. Nothing in this repository changed; the
platform underneath did.

---

## Timeline

| When (KST) | Event |
|---|---|
| Jul 13, evening | Full sdpa reproduction suite (12 runs × 24 episodes) completes on Colab L4. Results committed (`results_reproduction_sdpa/`, commit `567268d`). Rendering fully functional. |
| Jul 14, ~11:00 | New Colab session for SpatialVLA experiments. Setup succeeds; smoke test **segfaults** immediately after model load, at env creation. |
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
upstream; no fixed 2.x wheel exists, and SAPIEN 3.x is API-incompatible with
`ManiSkill2_real2sim`.

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

1. **Wait** for a Colab image fix/rollback or a SAPIEN-side fix. No timeline;
   SAPIEN 2.x is EOL so an upstream fix is unlikely.
2. **Rented GPU with a pinned driver image** (RunPod / Lambda / Vast):
   choose a CUDA 12.1 / driver 535–550 template and the problem does not
   exist. Estimated cost for the full remaining experiment plan: **$5–10**.
3. **Lab machine** with driver < 570 (the mentor's original local
   `run_experiment.sh` path works as-is).

## Preflight procedure for any future Colab session

Run **before** any setup; abandon the VM immediately on failure:

```bash
# 1) Driver check (5 s) — 580.x means the VM is unusable for SAPIEN 2.2
!nvidia-smi | grep "Driver Version"
```

```bash
# 2) Definitive render check (~2 min) if the driver looks OK
%%bash
apt-get install -y -qq python3.10 python3.10-venv > /dev/null 2>&1
python3.10 -m venv /content/vtest
/content/vtest/bin/pip install -q "numpy<2" sapien==2.2.2
export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
export LD_LIBRARY_PATH=/usr/lib64-nvidia
/content/vtest/bin/python - << 'EOF'
import sapien.core as sapien
e = sapien.Engine(); r = sapien.SapienRenderer(offscreen_only=True); e.set_renderer(r)
s = e.create_scene(); cam = s.add_camera('c', 128, 128, 1.0, 0.01, 10)
s.step(); s.update_render(); cam.take_picture()
print('RENDER OK -- VM is usable')
EOF
```

## What is NOT affected

- All committed reproduction results and reports (`results_reproduction_*`,
  `REPRODUCTION_REPORT.md`) — complete before the incident.
- The SpatialVLA foveation port (`SpatialVLA/experiments/tome/foveation.py`,
  `--foveate` flag) — code is committed and unit-tested; only *execution* of
  the simulation experiments is blocked.
- Model inference itself (CUDA path) — only Vulkan *rendering* is broken.

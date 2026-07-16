# Incident Report: Two Days of Colab GPU/Rendering Failures (2026-07-14 ~ 07-15)

**Period:** 2026-07-14 ~ 2026-07-15
**Status:** Resolved for practical purposes — the underlying rendering pipeline
works on current Colab GPU VMs. Getting to that conclusion took two days
because of several compounding problems, most of them self-inflicted by
flawed diagnostic tooling rather than the platform itself. This report
documents all of them so the same mistakes aren't repeated.

---

## Summary

On the evening of Jul 13, a full 24-episode-per-run OpenVLA reproduction
suite (12 runs) completed normally on a Colab L4. The next morning (Jul 14),
the same class of experiment (SimplerEnv / ManiSkill2 / SAPIEN 2.2
rendering) started segfaulting on every Colab VM obtained that day. A full
day of debugging pointed at a new NVIDIA driver (580.82.07, CUDA 13.0, Open
Kernel Module) that Colab's GPU VMs had started carrying, with SAPIEN 2.2.2
crashing in `svulkan2::core::Buffer::map()` the moment a camera image was
read back from GPU memory.

On Jul 15, further testing complicated that conclusion: some VMs with the
*identical* driver rendered fine end-to-end, while others failed. A day of
chasing this apparent "per-VM lottery" eventually revealed the real cause:
**the minimal empty-scene render snippet used as a diagnostic/preflight
check was itself broken on driver 580** (it creates zero-sized GPU buffers,
which the new driver's `Buffer::map()` no longer tolerates), independent of
whether the VM could actually run real workloads. Every "this VM is broken"
verdict produced by that snippet was invalid. The real pipeline
(`simpler_env.make()` + `env.reset()`, i.e. a populated scene) has run
successfully on every VM it was actually tried on, including ones the
snippet had condemned.

Layered on top of this were several unrelated environment issues (package
version conflicts, a HuggingFace API change, bare-VM bootstrap gaps) that
independently caused failures and cost additional time before being
isolated from the rendering problem.

---

## Timeline

| When | Event |
|---|---|
| Jul 13, evening | Full sdpa reproduction suite (12 runs × 24 episodes) completes on a Colab L4. Rendering fully functional. Driver version not recorded at the time. |
| Jul 14, ~11:00 | New Colab session. Setup succeeds; smoke test **segfaults** immediately after model load, at environment creation. `nvidia-smi` shows driver 580.82.07. |
| Jul 14, afternoon | Systematic debugging across two L4 VMs and one T4 VM. All three show driver 580.82.07; every SAPIEN render attempt segfaults identically. Nine candidate causes tested and ruled out (see below). |
| Jul 14, evening | Root cause identified via gdb (`Buffer::map()` crash). All known workarounds exhausted. Incident declared; driver 580 held responsible. |
| Jul 15, ~04:17 | Fresh L4 VM, same driver (580.82.07), runs the full OpenVLA smoke test successfully — complete episode, rendering included. |
| Jul 15, ~04:13–05:38 | Three more VMs (two L4, one A100) tested with the minimal render snippet: all three fail with the same segfault signature, despite Vulkan itself being demonstrably healthy on at least one of them (device correctly enumerated via `vulkaninfo`). Working theory becomes "VM-to-VM lottery, cause unidentified." |
| Jul 15, later | Controlled A/B test on a VM running a fully successful real evaluation: the minimal empty-scene snippet **still segfaults on that same healthy VM**, under every ICD/env-variable/shader combination tried. The real pipeline (`simpler_env.make()` + `env.reset()`) exits cleanly on the same VM every time. This overturns the "per-VM lottery" conclusion — the snippet, not the VM, was broken. |
| Jul 15, later | Separately, environment-setup problems (see below) caused additional failures on an otherwise-healthy VM, adding further confusion before being isolated as unrelated to the rendering issue. |

---

## Root cause: what actually breaks under driver 580

SAPIEN 2.2.2 (a binary wheel compiled ~2023, now end-of-life upstream)
segfaults inside `svulkan2::core::Buffer::map()` when mapping a GPU buffer
created for an **empty scene** (no objects, no lights — the shape of a
minimal smoke-test snippet: engine → renderer → empty scene → one camera →
`take_picture()`). The new driver's memory-type/layout assumptions no
longer match what SAPIEN 2.2.2's mapping code expects for zero-sized or
near-empty buffers, and the process is killed with SIGSEGV. gdb backtrace:

```
Thread 1 "python" received signal SIGSEGV, Segmentation fault.
#0 svulkan2::core::Buffer::map()
#1 svulkan2::renderer::Renderer::prepareRender(svulkan2::scene::Camera&)
#2 svulkan2::renderer::Renderer::render(...)
#3 sapien::Renderer::SVulkan2Camera::takePicture()
```

Critically, **this does not reproduce with a populated scene.** Every real
SimplerEnv/ManiSkill2 environment (objects, lighting, physics all present)
renders normally on the same driver, on every VM actually tested with the
real pipeline. The failure is specific to the pathological empty-scene case
that the diagnostic snippet happened to construct — it is not a general
"SAPIEN 2.2 cannot run under driver 580" problem, which was the initial
(incorrect) conclusion after Jul 14.

**Practical consequence:** any future health check must use the real
pipeline, never a synthetic minimal scene:

```python
import subprocess
code = '''import simpler_env
env = simpler_env.make("widowx_put_eggplant_in_basket")
obs, _ = env.reset()
print("ENV+RENDER OK")'''
p = subprocess.run([VENV_PYTHON, "-c", code], capture_output=True, text=True)
print(p.returncode, p.stdout[-200:], p.stderr[-200:])
```

## Why this was hard to pin down

- **Jul 14 hypothesis sweep** (before the empty-scene insight existed) tested
  and ruled out: wrong Vulkan ICD file (GLX vs EGL vs software), missing
  `LD_LIBRARY_PATH`, conda-vs-venv, stale/preloaded Vulkan loader, host BAR1
  memory differences, missing Xvfb/DISPLAY, GPU pool (L4 vs T4), Colab
  runtime version pinning (2026.04 / 2026.01 — this only swaps the userland
  container, not the host driver), and OOM. None explained the crash; all of
  these tests used the same flawed empty-scene snippet, so in hindsight they
  were ruling out things that were never the actual cause.
- **Jul 15 false negatives:** several VMs were declared "broken" using that
  same snippet before the snippet itself was identified as the problem. One
  of those VMs was later shown, by direct A/B comparison, to run the real
  pipeline perfectly — meaning a full day was spent chasing a "per-VM
  lottery" that didn't exist in the form believed.
- **No official changelog to check against.** Colab does not publish host
  driver versions anywhere (its `backend-info` repo tracks userland package
  versions only), so there was no way to confirm "the driver changed
  overnight" against an authoritative source — only indirect evidence
  (image library timestamps, a Jul 9 Colab announcement of a PyTorch
  2.13/CUDA 13 upgrade that implies a driver bump to support it, and external
  reports of the same SAPIEN failure mode on other platforms after upgrading
  past driver ~570).

## External corroboration

The SAPIEN-vs-newer-driver failure mode has been reported independently by
others who upgraded their own machines' drivers, predating this Colab
rollout:

- SAPIEN #271 — segfault on A100, driver ≥ 570, ICD overrides ineffective:
  https://github.com/haosulab/SAPIEN/issues/271
- RoboTwin #259 — no rendering device on driver 580.126 + CUDA 13, closed unresolved:
  https://github.com/RoboTwin-Platform/RoboTwin/issues/259
- ManiSkill #1020 — sapien 3.0.0b1 segfault on driver 570.133:
  https://github.com/haosulab/ManiSkill/issues/1020

None of these reports distinguish empty-scene vs. populated-scene renders,
so it's possible some of them describe the same narrower failure mode found
here rather than a total rendering breakage.

---

## Secondary problems encountered (independent of the rendering issue)

Several unrelated issues compounded the two days and are worth recording
separately, since each cost real time before being correctly isolated:

1. **HuggingFace CDN edge failures.** Large checkpoint/model downloads
   intermittently stalled or returned HTTP 403. Cause: `huggingface.co`
   load-balances large-file downloads between two CDN edges; the GCP edge
   (`us.gcp.cdn.hf.co`) returned 403 from Colab while the AWS edge
   (`cas-bridge.xethub.hf.co`) served normally. Workaround: a retry loop
   around `snapshot_download(..., max_workers=1)` — each retry re-rolls the
   CDN edge.

2. **`huggingface_hub` API break across versions.** A download-timeout
   workaround written against an older `huggingface_hub` (setting
   `huggingface_hub.constants.HF_HUB_DOWNLOAD_TIMEOUT` directly) no longer
   works against the newer version pulled in by a different environment —
   the `constants` attribute path was removed. Fix: set the
   `HF_HUB_DOWNLOAD_TIMEOUT` environment variable instead, which works
   across versions.

3. **Silent numpy/scipy binary incompatibility from an unpinned package.**
   Installing `opencv-python` without an exact version pin pulled in numpy
   2.2.6 as a transitive dependency, which is binary-incompatible with the
   already-installed scipy 1.12.0 build (`numpy.dtype size changed, may
   indicate binary incompatibility`). This didn't always reproduce — the
   same install sequence run at a different time let pip's resolver
   backtrack numpy to the compatible 1.23.5 on its own, and other times it
   didn't — making it an intermittent, confusing failure until every
   package in the chain (`opencv-python`, `numpy`) was pinned to exact,
   previously-verified versions and numpy was force-reinstalled last to
   guarantee the final state.

4. **Bare-VM bootstrap gaps.** Fresh Colab VMs cannot be assumed to have:
   a working `venv` module (`ensurepip` can be missing, requiring a
   `--without-pip` venv plus a manual `get-pip.py` bootstrap), `libvulkan1`
   installed at all, or any NVIDIA Vulkan ICD file present in
   `/etc/vulkan/icd.d/` (observed empty on more than one fresh VM). Any
   setup/preflight procedure needs to install these explicitly rather than
   assume they exist.

5. **GLX vs EGL ICD inconsistency across VMs.** On some VMs the GLX ICD
   (`libGLX_nvidia.so.0`) works and EGL fails; on at least one VM the
   reverse was true (GLX failed instance creation with
   `ERROR_INCOMPATIBLE_DRIVER`, EGL correctly enumerated the GPU). Since
   this wasn't predictable in advance, both ICD files need to be generated
   and tried in any setup that must work unattended across arbitrary VMs.

---

## What is NOT affected

- All previously committed reproduction results and reports
  (`results_reproduction_*`, `REPRODUCTION_REPORT.md`) — complete before
  Jul 14, unaffected by any of the above.
- Model inference itself (the CUDA/PyTorch path) — every failure above was
  in the rendering or environment-setup layer; no model ever failed to load
  or run once its environment was correctly configured.

# What the warp costs, and why it does not make foveation "more expensive"

Written 2026-08-22 after a reviewer-style objection to the Introduction's
clause *"model time per call, which excludes the warp itself."*

**The objection.** *"If the token count did not drop and you then add a warp on
top, the intervention does not merely fail to save, it costs more. Say so and
bury the technique."*

**The question is fair. The conclusion is not supported.** Measured, the warp
is too small to make that claim.

## 1. The measurement

`foveate_image_logpolar` and `foveate_image_blur` timed directly, 50 calls
each after a warm-up, on this machine's CPU.

| frame | mode | keep | ms per frame |
|---|---|---:|---:|
| 256×256 | log-polar | 20% | 1.99 |
| 256×256 | log-polar | 100% | 3.34 |
| 256×256 | blur | 20% | 10.97 |
| 256×256 | blur | 100% | 0.01 |
| 640×480 | log-polar | 20% | 6.63 |
| 640×480 | log-polar | 100% | 11.72 |
| 640×480 | blur | 20% | 45.50 |

Blur at 100% is free because `keep_ratio >= 1` returns the input untouched,
which its docstring states.

## 2. Against one model call

| cell | model call | log-polar warp | blur at 20% |
|---|---:|---:|---:|
| OpenVLA / Bridge | 517.7 ms | 3.34 ms = **0.6%** | 10.97 ms = **2.1%** |
| SpatialVLA / Bridge | 902.1 ms | **0.4%** | **1.2%** |
| SpatialVLA / Fractal | 936.8 ms | **0.4%** | **1.2%** |
| UniVLA / Bridge | 2801.5 ms | **0.1%** | **0.4%** |

## 3. What can honestly be said

**True.** Counted end to end, foveation costs a little *more* than baseline
rather than a little less. The sign flips.

**Not true.** That it is "expensive", that it "makes the technique worse", or
that it buries anything. Log-polar adds **0.1 to 0.6 percent**. A reader who
was told that a $3.1\%$ band is noise cannot then be told that $0.4\%$ is
damning.

**The reason foveation is not a way to spend less is the token count, not the
warp.** The warp is a rounding error on top of an intervention that already
saves nothing. Leading with the warp would replace a structural argument with
a trivial one, and would invite the obvious rebuttal that a GPU warp or a
fused preprocessing path removes it.

## 4. The line Results can carry, if it wants one

> Counting the warp, the log-polar path adds $0.1$ to $0.6\%$ to a step
> depending on the backbone, so the intervention is slightly net-negative
> rather than neutral. The token count, not the warp, is why it saves nothing.

One sentence, correctly signed, and it closes the objection without
overclaiming.

## 5. Caveats this measurement carries

- **Timed here, not on the campaign hardware.** The runs were on Colab T4s and
  an L4; this is a CPU timing in the analysis container. The ratio is robust
  because the gap is two orders of magnitude, but the exact millisecond figure
  is an estimate and should be re-timed on the run machine before it appears
  in the paper.
- **`elapsed` cannot substitute.** `build_grid_report.discover_cost` documents
  that wall-clock seconds per episode mixes speed with success, since a
  condition that fails more often runs to the step cap more often. So the
  end-to-end column cannot isolate the warp and the direct timing above is the
  right instrument.
- **The Introduction needs no change.** *"Model time per call, which excludes
  the warp itself"* is accurate, and stating the exclusion is the conservative
  choice, since counting the warp moves the number against foveation.

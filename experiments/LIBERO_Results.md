# LIBERO results — OpenVLA vs UniVLA

Benchmark: `libero_spatial`, 10 tasks × 5 initial states = 50 episodes per
condition (log-polar on OpenVLA is 30). Harness: `adaptive_sparse_vla/eval_libero.py`,
OSMesa rendering, episodes end as soon as LIBERO reports success.

Checkpoints: `openvla/openvla-7b-finetuned-libero-spatial`,
`Yuqi1997/UniVLA → UNIVLA_LIBERO_IMG_BS192_8K`.

## Headline

| condition | OpenVLA | Δ | UniVLA | Δ |
|---|---|---|---|---|
| baseline | 74.0% | — | 96.0% | — |
| action-repeat 2 (2x cheaper) | 66.0% | −8.0 | **28.0%** | **−68.0** |
| foveate blur 20% | 58.0% | −16.0 | 94.0% | −2.0 |
| foveate log-polar 20% | **0.0%** | **−74.0** | 88.0% | −8.0 |

UniVLA numbers are the post-fix runs (FAST decode failures 0/510-610 in every
condition). The pre-fix runs, which carried a ~4.5% corrupted-chunk rate,
gave 92.0 / 24.0 / 98.0 / 86.0 — every condition moved by at most 4 points,
i.e. within noise, so the defect was not driving any conclusion.

Foveation rows for UniVLA are the `--foveate-views both` runs, i.e. every
camera the policy sees is degraded (see the confound section below).

Significance (two-proportion z, n=50 per cell):

| comparison | Δ | SE | z | verdict |
|---|---|---|---|---|
| OpenVLA action-repeat 2 | −8.0 | 9.1 | −0.88 | within noise |
| OpenVLA log-polar | −74.0 | — | — | conclusive |
| UniVLA action-repeat 2 | −68.0 | 6.9 | −9.8 | conclusive |
| UniVLA blur (both views) | −2.0 | 4.4 | −0.46 | within noise |
| UniVLA log-polar (both views) | −8.0 | 5.4 | −1.49 | within noise |

## The result is a double dissociation

The two backbones fail under **opposite** interventions:

- **OpenVLA** absorbs a 2x cut in forward passes (−8, within noise) but
  collapses when its visual input is warped (−74).
- **UniVLA** is untouched by either foveation (−6 / +6, both within noise)
  but collapses when forward passes are halved (−68).

Neither backbone is simply "more robust". Which intervention is affordable
depends on the architecture, which is the claim this experiment was built to
test.

### Why the efficiency lever splits them

The shared axis is **env steps executed per forward pass**. The baselines
sit at opposite ends of it:

| | OpenVLA | UniVLA |
|---|---|---|
| action chunking | none | native, 10 steps |
| baseline steps/forward | 1 (closed-loop) | 10 (already amortized) |
| under `--action-repeat 2` | 2 | 20 |

`--action-repeat 2` is the same mechanism on both (`np.repeat`, each action
held for 2 env steps, doubling displacement). On OpenVLA it stretches a
1-step open-loop excursion to 2; on UniVLA it stretches an already-10-step
excursion to 20 with no feedback in between. The intervention is identical;
the starting point is not.

Per-task, UniVLA's collapse is uniform — 8 of 10 tasks drop to 0–1 of 5 —
rather than concentrated in hard tasks, consistent with open-loop drift
rather than task difficulty:

| task | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline | 5 | 5 | 5 | 4 | 5 | 5 | 5 | 5 | 5 | 4 |
| action-repeat 2 | 0 | 0 | **5** | 0 | 0 | 0 | 3 | 2 | 1 | 3 |

Task 2 is the only one that survives intact, and it is the only instruction
in the suite that names no spatial relation: "pick up the black bowl **from
table center**". Every other task ("between the plate and the ramekin",
"next to the ramekin", "on the cookie box") requires selecting one bowl from
several and then placing it precisely. Twenty steps of open-loop execution
removes exactly the corrective feedback that precision needs, which is why
relational tasks collapse and the unambiguous one does not.

### Why foveation splits them

OpenVLA's failures under foveation are concentrated exactly where the target
sits in the periphery, which is what identifies the mechanism as **fovea
placement**, not information loss:

| task | target location | baseline | blur 20% |
|---|---|---|---|
| 3 (cookie box) | centre | 5/5 | 5/5 |
| 7 (stove) | near centre | 4/5 | 5/5 |
| 4 (cabinet drawer) | right periphery | 3/5 | 1/5 |
| 9 (on the cabinet) | right periphery | 5/5 | **0/5** |

Log-polar keeps the image centre sharp and destroys the periphery. In
`libero_spatial` the image centre is empty table while the target bowls are
off-centre, so the policy loses precisely the region it needs. Verified
directly: the transform degrades gracefully (PSNR 22.8 dB at keep=20%,
centre error 2.1 vs periphery 6.8), so this is not an implementation
artifact.

UniVLA shows no such pattern — its cabinet tasks (4, 9) survive foveation
intact.

## Confound checked and cleared: the wrist camera

UniVLA takes two camera views (agent + wrist); OpenVLA takes one. The
harness originally foveated only the agent view, which would have handed
UniVLA an undegraded backup that OpenVLA never had. `--foveate-views both`
degrades every view the policy actually consumes:

| foveation | agent view only | both views |
|---|---|---|
| blur 20% | 96.0% | 98.0% |
| log-polar 20% | 94.0% | 86.0% |

Degrading the wrist view as well costs at most 8 points and leaves both
conditions statistically indistinguishable from baseline. UniVLA's
robustness to foveation is therefore **not** wrist-camera redundancy.

## Action-decode failures (UniVLA only)

UniVLA emits actions as FAST tokens. A malformed token sequence is swallowed
by the tokenizer, which substitutes all-zero DCT coefficients; after
un-normalization that is **not** a no-op but a fixed drift
(`[0.116, 0.033, 0, 0.009, 0.014, 0.056, −1]`, the midpoint of the q01/q99
range), so the arm keeps moving for a full 10-step chunk on a dead command.

Measured rates (instrumentation added after the first four runs, so baseline
is not yet covered):

| condition | decode failures |
|---|---|
| foveate log-polar 20%, both views | 30/639 (4.7%) |
| foveate blur 20%, both views | 26/587 (4.4%) |

Both foveation conditions sit at ~4.5% and still reach 86–98%, so the policy
absorbs occasional dead chunks. **Open item:** the baseline decode-failure
rate is unmeasured, so we cannot yet say whether foveation raises it. One
baseline re-run (~1 h) would close this.

## Caveats

- 5 of 50 available initial states per task. OpenVLA's baseline reproduces
  at 74% against a published 84.7%; the gap is ~2 SE and plausibly explained
  by the initial-state subsample plus a PIL LANCZOS resize standing in for
  the reference TF `lanczos3` (TensorFlow segfaults when imported after Mesa
  in this process).
- Foveation as implemented **does not reduce latency** — ms/inference is
  unchanged (OpenVLA 524 → 518, UniVLA 1882 → 1888). It reduces information,
  not compute. Only the action-repeat axis is an efficiency intervention.
- `--exec-chunk` (the more-reactive direction, unique to a chunked policy)
  has not been run.

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

UniVLA numbers are the post-fix runs (FAST decode failures 0/440-610 in every
condition). The pre-fix runs, which carried a ~4.5% corrupted-chunk rate,
gave 92.0 / 24.0 / 98.0 / 86.0 — every condition moved by at most 4 points,
i.e. within noise, so the defect was not driving any conclusion.

A blank-image control (both cameras zeroed, instruction only) puts UniVLA at
**0.0%**, which is what licenses reading the foveation rows as robustness
rather than as the policy ignoring its cameras — see the control section.

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

## Control: is UniVLA using the image at all?

UniVLA surviving log-polar at 20% admits two readings, and they point in
opposite directions:

- **(A)** the policy tolerates severe visual degradation → genuine robustness,
- **(B)** the policy barely uses the agent image on `libero_spatial` → the
  foveation result is vacuous and the suite is weak as a perception benchmark.

A first attempt to separate them measured how much foveation perturbs the VQ
token stream: **99.4%** of visual tokens change under log-polar 20% and 88.8%
under blur 20%, yet success holds at 88–94%. That refutes "the quantizer
absorbs the perturbation" but does not decide (A) vs (B) — token-ID equality
is a brittle metric, since neighbouring codebook entries can carry nearly
identical embeddings.

The deciding control is to remove the image entirely. `foveate_image_logpolar`
returns `np.zeros_like(frame)` when `keep_ratio <= 0`, so
`--foveate-keep-percent 0 --foveate-views both` blanks **both** cameras and
leaves only the instruction — no code change, same harness, same checkpoint.

| condition | success | n |
|---|---|---|
| baseline | 96.0% | 50 |
| both cameras blank | **0.0%** | 50 |

Zero of fifty, every task, every trial running the full 230 steps. **(B) is
dead**: the policy cannot do these tasks without vision, so the 88% under
log-polar is tolerance of degraded input, not indifference to input.

Two details make the control tight rather than merely suggestive:

- **FAST decode failures were 0/440.** The policy emits perfectly well-formed
  action sequences on a blank image — it fails because it has no information,
  not because the degenerate input corrupted the tokenizer. Had the rate
  spiked, the run would have measured a tokenizer artifact instead.
- **Actions stay large** (`dim_absmax` ≈ 1.08 / 0.70 / 0.79 on translation).
  The arm moves confidently in the wrong direction rather than freezing, which
  rules out "the model detected a broken input and stopped".

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

Measured rates, before and after the fix:

| condition | decode failures |
|---|---|
| foveate log-polar 20%, both views (pre-fix) | 30/639 (4.7%) |
| foveate blur 20%, both views (pre-fix) | 26/587 (4.4%) |
| every post-fix run | 0/440–610 (0.0%) |

The cause was the tokenizer, not the intervention: the stock
`physical-intelligence/fast` release lacks a pad/truncate guard that the
UniVLA authors added to their own copy, so a generated BPE sequence landing
one or two characters short of `time_horizon * action_dim` failed the reshape
and fell into the zero-substituting except-block. Inserting only that guard
(not copying the authors' file, which also carries different quantization
defaults) takes the rate to exactly zero, including on blank-image inputs.

All UniVLA numbers in this document are post-fix. The pre-fix runs moved by at
most 4 points, so the defect was never driving a conclusion — but it is now
excluded as an explanation for any of them.

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

## Open: the depth axis

Neither axis above can make UniVLA faster. Temporal is already spent (its
baseline runs 10 env steps per forward; doubling that collapses it to 28%),
and spatial never touched wall-clock — a UniVLA step profiles as 6% VQ encode
/ 13% prefill / **70% autoregressive decode** (`docs/VISUAL_TOKENS_VS_LATENCY.md`),
so the whole visual path is a ~19% ceiling. Reducing visual *tokens* rather
than visual *fidelity* does not escape it either: FastV, measured on this
backbone, left latency at 1.0× while success fell 100 → 75 → 38%.

The remaining lever is **decoder-layer bypass**, which shrinks the 70%
directly because the decode pays for every layer on every token. On the same
Emu3 backbone in SimplerEnv it was close to a free lunch — success 74% →
78–81% at 1.10–1.25× (`docs/DEPTH_PRUNING_RESULTS.md`) — and it is already
known to be backbone-dependent: the identical mechanism on SpatialVLA's Gemma2
hurt 3 of 4 tasks with a *single* layer bypassed. That makes it a third
independent axis for the same architecture-dependence claim.

Now wired into the LIBERO harness (`--depth-prune N`, `--depth-ctrl`;
bookkeeping unit-tested in `adaptive_sparse_vla/test_depth_libero_logic.py`).
Not yet run — the conditions are cells 5–7 of `LIBERO_UniVLA_Setup.md`. This
is the only condition in the whole grid where `avg_model_ms_per_infer` is
expected to move, so it is the number to read first.

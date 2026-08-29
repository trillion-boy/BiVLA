# Methodology for the six training-free interventions

This document is a paper-writing blueprint for **Bag of Tricks for
Training-Free Perception in Vision-Language-Action Models**. It describes the
implemented method families, their motivation, algorithms, calibration, and
claim boundaries. Numerical values below are implementation defaults, not
validated deployment thresholds. They must be selected on a calibration split
and frozen before final evaluation.

## 1. Problem formulation

At control step \(t\), base OpenVLA receives an RGB observation \(I_t\) and a
language instruction \(l\), then predicts a seven-dimensional action

\[
a_t = \pi_\theta(I_t,l), \qquad a_t \in \mathbb{R}^7,
\]

where the first six components encode translation and rotation and the final
component encodes the binary gripper decision. The model weights
\(\theta\) remain frozen for every intervention. Calibration selects only
inference-time layers, thresholds, budgets, and keyframe intervals; it never
updates model parameters.

The goal is to reduce end-to-end policy latency or the number of model calls
without reducing task success relative to optimized dense OpenVLA. An
intervention is considered Pareto-positive only if its latency confidence
interval improves and its paired task success passes a preregistered
non-inferiority or superiority test.

## 2. Method inventory

| family | intervention | level | role |
|---:|---|---|---|
| 1 | fixed foveation | observation | negative/robustness control |
| 2 | unconditional action repeat | control | negative temporal control |
| 3 | calibrated decoder-layer removal | model depth | positive candidate |
| 4 | guarded action reuse | control | positive candidate; evaluated as strict, moderate, and aggressive configurations |
| 5 | VLA-Cache | visual-token computation | recent baseline |
| 6 | interaction-aware temporal fusion and shared cache mask | visual representation and cache | primary positive candidate; evaluated as motion/entropy, task-aware, and conservative-adaptive configurations |

Optimized dense SDPA or FlashAttention is the reference system, not an
additional trick. The six families are evaluated separately before candidate
methods are combined.

## 3. Common dense reference

All methods use the same suite-specific base-OpenVLA checkpoint, language
prompt format, image resolution, action de-normalization key, simulator state,
and deterministic decoding. The reference executes all vision and language
layers using fused SDPA or FlashAttention under inference mode and BF16.

Latency measurement synchronizes CUDA immediately before and after policy
inference. We report median, mean, and p95 end-to-end policy latency, model
calls per episode, control frequency, and peak memory. Eager attention is used
only when a baseline requires explicit attention maps and is never the main
dense timing reference.

## 4. Trick 1: fixed foveation

### Motivation

Robot manipulation often concentrates task-relevant content near the object
and gripper. Fixed foveation tests whether preserving a sharp central region
while suppressing peripheral detail improves robustness to clutter. Because
the output resolution and token count do not change, it is a negative control
for the claim that perceptual simplification alone causes acceleration.

### Transformation

For an image of height \(H\) and width \(W\), a keep ratio \(\rho\) defines
the sharp-disc radius

\[
r_s = \sqrt{\frac{\rho HW}{\pi}}.
\]

Pixels inside \(r_s\) are copied exactly. Outside the disc, distance-dependent
weights blend the original image with Gaussian-blurred images using standard
deviations 3 and 9 pixels. The implementation default is \(\rho=0.20\), with
the foveal center at the image center. Geometry, spatial resolution, and the
256-patch OpenVLA token sequence remain unchanged.

### Experimental role

Evaluate the same fixed transform on every frame and report both success and
preprocessing-inclusive latency. A result with unchanged model latency is
expected and scientifically useful. This condition must not be advertised as
an acceleration contribution.

## 5. Trick 2: unconditional action repeat

### Motivation

Adjacent observations frequently yield similar actions. Fixed repetition is
the simplest way to exploit this temporal redundancy and establishes the
failure mode of skipping feedback without a safety gate.

### Controller

Given a dense policy action \(a_t\) and repeat factor \(k\), the controller
executes

\[
\tilde a_{t+j}=a_t, \qquad j=0,\ldots,k-1,
\]

and does not query the policy during the repeated steps. The default negative
control uses \(k=2\). It approximately halves calls per environment step but
doubles the interval over which the robot acts without fresh policy feedback.

### Experimental role

Report success, model calls, contact failures, and gripper errors. This method
is a negative temporal control, not a candidate contribution. It should never
be included in the final combined positive system.

## 6. Trick 3: calibrated decoder-layer removal

### Motivation

Residual decoder blocks whose outputs closely match their inputs may be
redundant for the target observation-instruction distribution. Removing a
small number of such blocks reduces both prefill and autoregressive decoding
work without retraining.

### Calibration score

For decoder block \(j\), let \(H_{j,n}^{\mathrm{in}}\) and
\(H_{j,n}^{\mathrm{out}}\) be its input and output hidden states for calibration
sample \(n\). We compute the ShortGPT-style block influence

\[
s_j = \frac{1}{N}\sum_{n=1}^{N}
\left[1-\operatorname{mean}_{q}
\cos\!\left(H_{j,n,q}^{\mathrm{in}},H_{j,n,q}^{\mathrm{out}}\right)\right],
\]

where \(q\) indexes sequence positions. Low \(s_j\) indicates that a block
perturbs its input relatively little.

### Safeguarded selection and execution

Layers are ranked once on the disjoint calibration split. The selector:

1. protects the first 25% of decoder blocks;
2. protects the final decoder block;
3. selects the lowest-influence remaining blocks;
4. requires at least one retained block between any two removed blocks; and
5. freezes the chosen indices before validation and testing.

Selected modules are structurally removed, retained attention layers are
reindexed, and the configured decoder depth is updated so the generation
cache remains valid. This is true removal rather than an identity block that
still allocates placeholder KV entries.

### Operating points and safeguards

Evaluate removal budgets of 0, 1, 2, and 4 of 32 blocks, but treat one block
as the primary conservative point unless validation supports more. Report
gripper disagreement separately from continuous-action L2 because the latter
can hide a binary gripper flip. Layer identities and calibration scores must
be included in the supplementary material.

## 7. Trick 4: guarded action reuse

### Motivation

Fixed repeat saves complete model calls but is unsafe near contacts or scene
changes. Guarded reuse makes the same saving conditional on agreement in both
the current visual observation and the recent policy trajectory.

### Visual stability tests

The controller samples every eighth image pixel to obtain a cheap signature
\(S_t\). It requires whole-frame stability

\[
d_{\mathrm{global}}=\operatorname{mean}|S_t-S_{t-1}|
\le \tau_g,
\]

and local stability

\[
d_{\mathrm{local}}=\max_{p\in\mathcal P}
\operatorname{mean}_{x\in p}|S_t(x)-S_{t-1}(x)|
\le \tau_l,
\]

where \(\mathcal P\) is an \(8\times8\) grid. The defaults are
\(\tau_g=0.01\) and \(\tau_l=0.03\) for RGB values normalized to \([0,1]\).
The maximum-patch statistic prevents a small gripper or object change from
being diluted by a static background.

### Action stability tests

At least two actions must have been produced by dense inference. For the two
most recent dense actions \(a_{t-2}\) and \(a_{t-1}\), reuse is allowed only
when

\[
\cos(a_{t-2}^{0:6},a_{t-1}^{0:6}) \ge 0.995,
\]

the current translation norm is at least 0.01, and the signs of the gripper
components agree. The lower bound on translation deliberately disables reuse
during low-motion precision phases. Reuse is capped at one consecutive step.

### Decision rule

If every visual and action gate passes, the controller returns the previous
action without calling OpenVLA. Otherwise it performs dense inference and
updates the two-action history. Thus every step still receives a current RGB
observation even when model execution is skipped.

The synthetic latency script sets the translation threshold to zero only to
exercise the reuse path. That diagnostic override is not the proposed rollout
configuration.

### Evaluation

Besides success and latency, report the reuse rate, fallback rate, calls per
episode, maximum consecutive reuse, and reuse frequency during reach, contact,
grasp, and placement phases. All thresholds must be calibrated and frozen.

## 8. Trick 5: VLA-Cache baseline

### Motivation

Consecutive robot observations contain many visually static patches.
VLA-Cache accelerates inference by reusing computation for static patches
while recomputing dynamic or task-relevant patches. Historical token
information is retained, unlike token pruning.

### Patch and task selection

For consecutive frames, VLA-Cache first ranks visual patches by image change
and produces a static candidate set. Task relevance is computed from a
selected language-model layer's attention tensor. Attention is averaged over
heads, text-query rows are averaged for each visual key, and the top-ranked
visual patches are protected. Static patches outside the protected set are
translated into their visual-token sequence positions and marked reusable.

The maintained helper uses layer 15 by default, assumes 256 visual tokens
beginning at sequence position 1, and corrects the released helper's
incompatible mask dimensionality. The official layer schedule then determines
which decoder layers use cached rather than recomputed visual KV states.

### Calibration and comparison

Sweep the number of candidate static patches only on calibration/validation
episodes and freeze one operating point. Compare steady-state cached latency
with optimized dense inference, while separately reporting first-frame cache
construction. Report actual reusable patches rather than only the requested
budget. Because explicit attention maps can force eager attention, include an
honest end-to-end comparison and disclose the backend used by each path.

VLA-Cache is a literature baseline. Changes needed to reproduce it should not
be presented as a new first-party trick.

## 9. Trick 6: interaction-aware temporal fusion and shared cache mask

### Motivation

Temporal fusion can stabilize representations, while selective cache reuse
can reduce computation. Using a shared conservative decision for both avoids
a failure mode in which the denoising path treats a patch as stable while the
acceleration path independently caches a task-critical region.

### Patch-level signals

Each frame is divided into the \(16\times16\) grid corresponding to OpenVLA's
256 visual patches. For patch \(i\), the selector computes:

- motion \(m_i\): normalized RGB mean absolute difference between consecutive
  frames;
- entropy \(e_i\): normalized 16-bin grayscale histogram entropy; and
- task relevance \(r_i\): optional text-query-to-vision attention, normalized
  within the frame.

A patch is protected when \(m_i>0.01\), when it lies in the highest-entropy
15%, when it lies in the most task-relevant 20%, or when an external
interaction detector explicitly protects it. Protection is dilated by one
neighboring patch in every spatial direction. This conservative dilation
keeps the vicinity of objects and contacts dense.

Among the remaining candidates, patches are ranked by

\[
R_i=m_i+e_i+r_i,
\]

with deterministic patch-index tie breaking. At most 50% of all patches are
declared reusable. The selector records reusable indices, protected indices,
and all three signals so every decision can be audited.

### Temporal token fusion

Let \(z_{t,i}\) denote the current projected visual token and
\(\tilde z_{t-1,i}\) the representation retained from the previous step. For
the reusable set \(\mathcal R_t\), the fused representation is

\[
\tilde z_{t,i}=
\begin{cases}
\tilde z_{t-1,i}, & i\in\mathcal R_t,\\
z_{t,i}, & i\notin\mathcal R_t.
\end{cases}
\]

Every third frame is a dense keyframe by default, and a keyframe can be forced
after uncertainty or a control event. The retained representation is the
previous fused output, so periodic keyframes bound propagation length.

Fusion is an accuracy/temporal-consistency candidate. It does not skip the
vision encoder, projector, or decoder and therefore is not itself a speedup.

### Shared-mask cache acceleration

The same reusable patch indices are shifted to their OpenVLA sequence
positions and exposed to the VLA-Cache-compatible path. The intended combined
method reuses visual KV/KQV computation only for \(\mathcal R_t\), while all
protected patches are recomputed. This is the speed mechanism paired with the
fusion hypothesis.

The shared-mask interface is implemented, but its selective KV/KQV execution
is not yet integrated into the main optimized SDPA path. Until that integration
and rollout evaluation are complete, describe this part as a candidate method,
not a measured acceleration result.

### Required ablations

Evaluate:

1. dense OpenVLA;
2. fusion without caching;
3. motion-only selection;
4. motion plus entropy;
5. motion plus task relevance;
6. the full selector;
7. full selector without dilation;
8. full selector without periodic keyframes;
9. cache reuse without token fusion; and
10. shared-mask fusion plus cache reuse.

Report protected/reused patch fractions, selector overhead, dense-keyframe
rate, action and gripper disagreement, task success, and end-to-end latency.

## 10. Combined positive system

Only methods that independently pass validation gates enter the combined
system. The proposed execution order is:

1. observe the current RGB frame;
2. allow guarded action reuse to skip the model only if every control-level
   safeguard passes;
3. otherwise form the interaction-aware patch mask;
4. apply shared-mask fusion and selective cache reuse;
5. execute the calibrated reduced decoder, if depth removal independently
   passed validation; and
6. decode and execute the new action.

Fixed foveation and unconditional action repeat are excluded from this system.
VLA-Cache alone remains a baseline condition. A combined model must be tested
directly because individual non-inferiority does not imply that approximations
compose safely.

## 11. Calibration and evaluation protocol

Use disjoint calibration, validation, and test splits. Calibration chooses
layer indices, thresholds, patch budgets, protected fractions, and keyframe
intervals without updating model weights. Validation selects one frozen
operating point per candidate and one combined setting. Test episodes use
identical task seeds and initial states across conditions.

Use suite-specific base-OpenVLA checkpoints for LIBERO-Spatial,
LIBERO-Object, LIBERO-Goal, and LIBERO-Long. For every condition report:

- paired task success with confidence intervals;
- median, mean, and p95 policy latency with CUDA synchronization;
- end-to-end control frequency and model calls per episode;
- action disagreement and per-dimension deviations;
- binary gripper disagreement;
- reuse, fallback, and keyframe rates by task phase; and
- peak GPU memory and preprocessing/cache overhead.

A literal accuracy-improvement claim requires a success superiority test. A
“no accuracy loss” claim requires a preregistered non-inferiority margin. The
detailed gates and minimum scale are specified in
`docs/experiment_protocol.md`.

## 12. Reproducibility and claim boundaries

All canonical measurements must be emitted as JSON under
`artifacts/results/`. Record the checkpoint revision, normalization key,
attention backend, BF16/TF32 configuration, simulator commit, render settings,
task seed, and every calibrated threshold.

Single-image action agreement, aggregate action L2, and synthetic-frame
latency are compatibility diagnostics. They are never reported as robot-task
accuracy or safety evidence. A method enters the paper's positive bag only
after paired rollouts show a latency gain and success non-inferiority or
superiority against optimized dense inference.

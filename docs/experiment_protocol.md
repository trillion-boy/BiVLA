# Evaluation protocol for a defensible ICRA claim

## Claim gates

A single-image action comparison establishes compatibility, not accuracy. A
method enters the positive bag only if, on untouched test episodes, it passes:

1. **Speed superiority:** the paired/bootstrap 95% confidence interval for
   end-to-end policy latency is entirely below the dense optimized baseline.
2. **Success superiority** for a literal "improves accuracy" claim, or a
   preregistered non-inferiority margin (suggested: -1 percentage point) for a
   "no accuracy loss" claim.
3. **Safety/precision checks:** no regression on contact-heavy and long-horizon
   strata, not merely an improved macro average.

Do not call a one- or two-point change an improvement without its interval.

## Split the work correctly

- **Calibration split:** choose depth layers and all cache/reuse thresholds.
  Labels and rewards are not needed for block influence, but calibration
  episodes must never be reused for the final success estimate.
- **Validation split:** select one operating point per trick and one combined
  operating point. Freeze them before the test run.
- **Test split:** identical task/initial-state seeds for every condition.
  Never retune after viewing these outcomes.

This remains training-free: no weights or learned parameters are updated.

## Conditions

Run at least:

1. eager dense baseline (diagnostic only);
2. optimized dense baseline (SDPA/FlashAttention, inference mode);
3. fixed foveation and fixed repeat negative controls;
4. temporal token fusion alone (accuracy/denoising mechanism);
5. VLA-Cache alone;
6. shared interaction-aware fusion + cache mask;
7. conservative depth alone (0, 1, 2, 4 blocks);
8. conservative action reuse alone;
9. positive tricks combined;
10. oracle/fallback ablations that expose how often each safeguard triggers.

For base OpenVLA, use the official fine-tuned checkpoints for all four LIBERO
suites, not the Bridge checkpoint with LIBERO normalization. Record the
checkpoint commit, `unnorm_key`, gripper transform, simulator commit, task
seed, and render configuration.

## Metrics

- success with Wilson interval and paired McNemar comparison;
- median, p95, and mean policy latency with explicit CUDA synchronization;
- environment steps per model call, calls per episode, and total model time;
- end-to-end control frequency, not only isolated kernel time;
- peak VRAM, FLOPs (secondary), and CPU preprocessing/cache overhead;
- action disagreement rate and per-dimension deviation from dense OpenVLA;
- reuse/fallback rate stratified by reach, contact, grasp, and placement phase.

Report speed–success curves, not a cherry-picked point. Correct multiple
condition comparisons (for example, Holm correction), and show bootstrap
confidence regions on the Pareto plot.

## Minimum statistical scale

The existing 24–50 episode summaries are too coarse for small differences.
At 75% success, a rough independent-binomial 95% margin is about 8.5 points at
100 episodes and 3.8 points at 500 episodes. Pairing helps, but does not make a
1-point gain automatically credible. Use the standard 50 trials x 10 tasks per
LIBERO suite where feasible and preserve the per-episode paired outcomes.

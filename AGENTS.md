# Agent operating guide

## Mission

Evaluate training-free methods for base OpenVLA that improve end-to-end policy
latency without reducing robot-task success. Synthetic action agreement is a
compatibility diagnostic, never an accuracy result.

## First five minutes

1. Read `README.md`.
2. Read `docs/local_results.md` before proposing or rerunning a method.
3. Run `make doctor` and `make quick`.
4. Use code in `src/vla_tricks/`; notebooks are explanatory snapshots.
5. Put generated JSON in `artifacts/results/`.

## Canonical commands

```bash
make doctor
make quick
make benchmark
make cache
```

`make all` runs everything above and takes roughly one minute after the model
is cached. Full LIBERO rollouts are intentionally not part of `make all`.

## Scientific guardrails

- Compare against optimized dense SDPA/FlashAttention, not an artificially
  slow eager baseline.
- Use suite-specific OpenVLA LIBERO checkpoints for task-success claims.
- Calibrate thresholds/layers on a disjoint split and freeze before testing.
- Report paired success, median/p95 latency, model calls, action disagreement,
  gripper disagreement, and confidence intervals.
- Fixed foveation and fixed action repeat are negative controls.
- One-layer depth removal, guarded reuse, and cache methods are candidates,
  not validated positive results.
- Do not infer safety from aggregate action L2: a two-layer local test flipped
  the binary gripper decision.

## Code ownership

- First-party edits belong in `src/`, `scripts/`, `tests/`, `docs/`, or
  `configs/`.
- Treat `third_party/` as vendored code. Its three intentional LIBERO
  compatibility edits are listed in `third_party/README.md`.
- Never run recursive pytest from `third_party/transformers-vla-cache`; the
  upstream suite requires many unrelated development dependencies.
- Do not edit notebook outputs to manufacture results. Canonical measurements
  are JSON artifacts produced by scripts.
- Do not download additional 7B checkpoints unless the task requires them;
  each suite-specific checkpoint is large.

## Important paths

- Environment Python: the active `vla_tricks` environment Python
- Base model: `models/openvla-7b`
- LIBERO config: `configs/libero/config.yaml`
- Results: `artifacts/results/`
- Full evaluation entry point:
  `third_party/vla-cache/src/openvla/experiments/robot/libero/run_libero_eval.py`

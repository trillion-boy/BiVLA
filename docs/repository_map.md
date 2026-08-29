# Repository map and execution flow

## Fast path

```text
configs/environment.yml
          |
          v
scripts/activate_env.sh -----> make doctor
                                  |
                                  v
                       make quick (tests + LIBERO)
                                  |
                     +------------+-------------+
                     v                          v
              make benchmark               make cache
                     |                          |
                     +------------+-------------+
                                  v
                 artifacts/results/<model>/<config>/
                                  |
                                  v
                      docs/local_results.md
```

The scripts, not the notebooks, are the executable source of truth.

## What belongs where

- `src/vla_tricks/`: small reusable mechanisms with no environment-specific
  orchestration.
- `scripts/`: executable orchestration. Every script must work from the
  repository root and expose `--help` when it accepts options.
- `scripts/experiments/`: the 14 independent rollout configuration entry
  points. Each fixes one condition, all thresholds, the checkpoint, and its
  canonical result directory.
- `tests/`: CPU-fast tests for selection, gating, restoration, and shape
  contracts.
- `configs/`: pinned environment and local simulator paths.
- `docs/`: decisions a new contributor must know before experimentation.
- `artifacts/results/<model>/<config>/`: machine-readable benchmark evidence;
  each independent configuration owns one result directory.
- `third_party/`: external repositories, isolated so search/test tools do not
  confuse them with first-party code.

## Full paper evaluation

The quick path uses the base Bridge OpenVLA checkpoint and synthetic images.
For paper accuracy, download each official suite-specific checkpoint and run
paired LIBERO trials following `docs/experiment_protocol.md`. Use the evaluator
under `third_party/vla-cache/src/openvla/experiments/robot/libero/`, but write
new first-party wrappers and summaries outside `third_party/`.

# Shareable project bundle

This directory is a portable copy of the maintained project code, configs,
documentation, tests, and generated JSON results. Large model checkpoints and
vendored dependencies are intentionally omitted.

## Setup

From this directory:

```bash
bash third_party/setup_third_party.sh
bash models/setup_models.sh
bash scripts/setup_env.sh
make quick
```

`models/setup_models.sh` downloads the public OpenVLA checkpoint by default.
The MiniVLA checkpoints used by the included SimplerEnv experiment entry
points are not redistributed here; provide those checkpoints separately and
place them under `models/`.

The bundle contains no robot data or checkpoint weights. See `docs/` and
`artifacts/results/` for the experiment protocol and existing measurements.

# SimplerEnv Subtree — Runtime Dependency Only

This vendored `SimplerEnv/` subtree is kept only as a runtime dependency for:
- the **UniVLA baseline** evaluation
- the **UniVLA + PostNorm** evaluation

The evaluation code in this repository imports:
- `simpler_env/__init__.py`
- `simpler_env/utils/env/env_builder.py`
- `simpler_env/utils/env/observation_utils.py`
- `ManiSkill2_real2sim/` assets and environment implementation

Policy runners, debugging helpers, and metrics scripts were removed from the main repository surface.

#!/usr/bin/env python3
"""Independent strict guarded-action-reuse configuration."""
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "scripts" / "run_minivla_simpler_eval.py"
sys.argv = [str(RUNNER), "--checkpoint", str(ROOT / "models/minivla-vq-bridge-prismatic"), "--condition", "guarded_action_reuse", "--model-name", "minivla_simplerenv", "--config-name", "guarded_reuse_strict", "--output-dir", str(ROOT / "artifacts/results/minivla_simplerenv/guarded_reuse_strict"), "--num-trials-per-task", "50", "--seed", "42", "--max-steps", "150", "--device", "cuda:0", "--reuse-max-frame-mae", "0.01", "--reuse-max-local-patch-mae", "0.03", "--reuse-min-action-cosine", "0.995", "--reuse-min-translation-norm", "0.01", "--reuse-max-consecutive", "1"]
runpy.run_path(str(RUNNER), run_name="__main__")

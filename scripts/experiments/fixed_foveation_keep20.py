#!/usr/bin/env python3
"""Independent fixed-foveation keep-20-percent configuration."""
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "scripts" / "run_minivla_simpler_eval.py"
sys.argv = [str(RUNNER), "--checkpoint", str(ROOT / "models/minivla-vq-bridge-prismatic"), "--condition", "fixed_foveation", "--model-name", "minivla_simplerenv", "--config-name", "fixed_foveation_keep20", "--output-dir", str(ROOT / "artifacts/results/minivla_simplerenv/fixed_foveation_keep20"), "--num-trials-per-task", "50", "--seed", "42", "--max-steps", "150", "--device", "cuda:0", "--fovea-keep-ratio", "0.20"]
runpy.run_path(str(RUNNER), run_name="__main__")

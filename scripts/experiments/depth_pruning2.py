#!/usr/bin/env python3
"""Independent two-layer decoder-pruning configuration."""
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "scripts" / "run_minivla_simpler_eval.py"
sys.argv = [str(RUNNER), "--checkpoint", str(ROOT / "models/minivla-vq-bridge-prismatic"), "--condition", "depth_pruning", "--model-name", "minivla_simplerenv", "--config-name", "depth_pruning2", "--output-dir", str(ROOT / "artifacts/results/minivla_simplerenv/depth_pruning2"), "--num-trials-per-task", "50", "--seed", "42", "--max-steps", "150", "--device", "cuda:0", "--depth-layers", "2"]
runpy.run_path(str(RUNNER), run_name="__main__")

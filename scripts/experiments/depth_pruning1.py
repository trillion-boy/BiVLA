#!/usr/bin/env python3
"""Independent one-layer decoder-pruning configuration."""
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "scripts" / "run_minivla_simpler_eval.py"
sys.argv = [str(RUNNER), "--checkpoint", str(ROOT / "models/minivla-vq-bridge-prismatic"), "--condition", "depth_pruning", "--model-name", "minivla_simplerenv", "--config-name", "depth_pruning1", "--output-dir", str(ROOT / "artifacts/results/minivla_simplerenv/depth_pruning1"), "--num-trials-per-task", "50", "--seed", "42", "--max-steps", "150", "--device", "cuda:0", "--depth-layers", "1"]
runpy.run_path(str(RUNNER), run_name="__main__")

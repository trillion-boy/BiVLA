#!/usr/bin/env python3
"""Independent motion-plus-entropy temporal-fusion configuration."""
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "scripts" / "run_minivla_simpler_eval.py"
sys.argv = [str(RUNNER), "--checkpoint", str(ROOT / "models/minivla-vq-bridge-prismatic"), "--condition", "temporal_fusion", "--model-name", "minivla_simplerenv", "--config-name", "temporal_fusion_motion_entropy", "--output-dir", str(ROOT / "artifacts/results/minivla_simplerenv/temporal_fusion_motion_entropy"), "--num-trials-per-task", "50", "--seed", "42", "--max-steps", "150", "--device", "cuda:0", "--fusion-keyframe-interval", "3", "--fusion-motion-threshold", "0.01", "--fusion-entropy-protect-fraction", "0.15", "--fusion-task-protect-fraction", "0.20", "--fusion-protect-radius", "1", "--fusion-max-reuse-fraction", "0.50"]
runpy.run_path(str(RUNNER), run_name="__main__")

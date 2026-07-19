"""
Per-module latency breakdown for the RoboVLMs SimplerEnv eval.

Mirrors the profiling done earlier on OpenVLA/SpatialVLA (where autoregressive
decode dominated). RoboVLMs KosMos has a different inference shape -- a single
non-autoregressive forward (vision encoder -> vision-to-text projection ->
Kosmos2 text transformer) feeding a small stateful LSTM action head -- so the
interesting question is which of those stages owns the ~75 ms/step.

Usage: `--profile-latency` on eval/simpler/main_inference.py. Instance-level
`forward` wrappers time each stage with CUDA synchronization (so GPU kernels
are attributed to the module that launched them), and the whole
`model.step()` is timed separately; the gap between the step total and the
sum of the stages is reported as "other" (image preprocess, tokenization,
CPU<->GPU transfers, action un-normalization and glue code).
"""

import time
from collections import OrderedDict

import torch


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _resolve(root, dotted):
    obj = root
    for part in dotted.split("."):
        obj = getattr(obj, part, None)
        if obj is None:
            return None
    return obj


class LatencyProfiler:
    def __init__(self):
        self.totals = OrderedDict()
        self.calls = OrderedDict()
        self.step_total = 0.0
        self.step_calls = 0

    def _record(self, name, dt):
        self.totals[name] = self.totals.get(name, 0.0) + dt
        self.calls[name] = self.calls.get(name, 0) + 1

    def wrap_module(self, module, name):
        if module is None:
            print(f"[Profiler] module '{name}' not found; skipping")
            return
        self.totals.setdefault(name, 0.0)
        self.calls.setdefault(name, 0)
        orig = module.forward

        def timed_forward(*args, **kwargs):
            _sync()
            t0 = time.perf_counter()
            out = orig(*args, **kwargs)
            _sync()
            self._record(name, time.perf_counter() - t0)
            return out

        module.forward = timed_forward

    def wrap_step(self, model):
        orig = model.step

        def timed_step(*args, **kwargs):
            _sync()
            t0 = time.perf_counter()
            out = orig(*args, **kwargs)
            _sync()
            self.step_total += time.perf_counter() - t0
            self.step_calls += 1
            return out

        model.step = timed_step

    def print_summary(self):
        if not self.step_calls:
            print("[Profiler] no model.step() calls recorded")
            return
        step_ms = self.step_total / self.step_calls * 1000.0
        print("\n=== Latency breakdown (per env step, CUDA-synchronized) ===")
        print(f"model.step() total: {step_ms:.1f} ms over {self.step_calls} steps")
        accounted = 0.0
        for name, total in self.totals.items():
            calls = self.calls[name]
            per_step_ms = total / self.step_calls * 1000.0
            per_call_ms = (total / calls * 1000.0) if calls else 0.0
            pct = 100.0 * total / self.step_total if self.step_total else 0.0
            accounted += total
            print(
                f"  {name:<34} {per_step_ms:7.1f} ms/step "
                f"({pct:5.1f}%)  [{calls} calls, {per_call_ms:.1f} ms/call]"
            )
        other = self.step_total - accounted
        other_ms = other / self.step_calls * 1000.0
        other_pct = 100.0 * other / self.step_total if self.step_total else 0.0
        print(
            f"  {'other (preprocess/tokenize/glue)':<34} {other_ms:7.1f} ms/step "
            f"({other_pct:5.1f}%)"
        )
        print("=" * 59)


def install_profiler(model):
    """model: BaseModelInference. Wraps the KosMos stages + model.step()."""
    prof = LatencyProfiler()
    robovlm = _resolve(model, "model.model")  # BaseTrainer.model -> RoboKosMos
    targets = [
        ("vision encoder (CLIP ViT-L)", "backbone.vision_model"),
        ("vision->text projection", "backbone.image_to_text_projection"),
        ("LLM transformer (Kosmos2 text)", "backbone.text_model"),
        ("action head (LSTM)", "act_head"),
    ]
    for name, path in targets:
        prof.wrap_module(_resolve(robovlm, path), name)
    prof.wrap_step(model)
    print("[Profiler] installed; per-module latency will print after the run.")
    return prof

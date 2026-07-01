"""
Training-free depth (layer) pruning for SpatialVLA's Gemma2 decoder.

This is the *latency* lever for SpatialVLA, mirroring what we validated on
UniVLA/Emu3: the per-step cost is decode-bound (~75%), so the only thing that
shrinks wall-clock is making each of the many sequential action-token forwards
cheaper -- i.e. bypassing redundant decoder layers.

Why this is safe on Gemma2 (unlike Emu3/FastV):
  * We bypass LAYERS, not tokens -> token positions are untouched, so RoPE
    spatial grounding is preserved (no re-indexing bug).
  * Gemma2 generation uses HybridCache, which pre-allocates a slot per layer.
    A bypassed layer simply never writes/reads its slot -> no DynamicCache gap
    / IndexError. The layer loop `for l in self.layers[:num_hidden_layers]`
    reads only `layer_outputs[0]` (hidden_states) during generation, so a
    bypass that returns `(hidden_states,)` is a correct drop-in.

Redundancy = cos(layer_input, layer_output): a layer whose output ≈ its input
barely changes the residual stream, so it is the safest to drop. We calibrate
this ranking ONCE (one forward, via hooks), then bypass the top-N most redundant
layers -- training-free, no data, no fine-tuning.

Usage:
    pruner = DepthPruner(policy.vla.language_model)
    pruner.install_calibration_hooks()
    policy.step(image, instruction)          # one real step to populate stats
    pruner.finalize_calibration()
    pruner.apply(count=4, min_layer=2)        # bypass 4 most-redundant layers
    ...
    pruner.restore()
"""
from __future__ import annotations

import torch
import torch.nn as nn


class _BypassLayer(nn.Module):
    """Returns the hidden state unchanged. Keeps the original layer for restore.

    Matches Gemma2DecoderLayer's generation-time contract: the model loop reads
    only `layer_outputs[0]`, and (with HybridCache) does not require this layer
    to touch the cache."""

    def __init__(self, orig: nn.Module):
        super().__init__()
        self.orig = orig

    def forward(self, hidden_states, *args, **kwargs):
        return (hidden_states,)


def _get_layers(language_model) -> nn.ModuleList:
    # language_model = Gemma2ForCausalLM ; .model = Gemma2Model ; .layers = ModuleList
    return language_model.model.layers


class DepthPruner:
    def __init__(self, language_model):
        self.lm = language_model
        self.layers = _get_layers(language_model)
        self.n = len(self.layers)
        self.orig = list(self.layers)          # references to real layers
        self._sum = [0.0] * self.n             # accumulated cos-sim
        self._cnt = [0] * self.n
        self._handles = []
        self.ranking = None                    # most -> least redundant (layer idx)
        self.pruned = []

    # ---- calibration ------------------------------------------------------
    def install_calibration_hooks(self):
        """Register forward hooks that accumulate cos(input, output) per layer.
        Run one (or a few) real forward passes after calling this."""
        self._sum = [0.0] * self.n
        self._cnt = [0] * self.n

        def mk(i):
            def hook(module, inp, out):
                hi = inp[0]
                ho = out[0] if isinstance(out, tuple) else out
                a = hi.reshape(-1, hi.shape[-1]).float()
                b = ho.reshape(-1, ho.shape[-1]).float()
                cs = torch.nn.functional.cosine_similarity(a, b, dim=-1).mean().item()
                self._sum[i] += cs
                self._cnt[i] += 1
            return hook

        for i, layer in enumerate(self.orig):
            self._handles.append(layer.register_forward_hook(mk(i)))

    def finalize_calibration(self):
        """Compute the redundancy ranking and remove the hooks."""
        for h in self._handles:
            h.remove()
        self._handles = []
        red = [(self._sum[i] / self._cnt[i]) if self._cnt[i] else -1.0
               for i in range(self.n)]
        # higher cos = more redundant = safer to drop; rank descending
        self.ranking = sorted(range(self.n), key=lambda i: red[i], reverse=True)
        self._redundancy = red
        return self.ranking

    # ---- apply / restore --------------------------------------------------
    def apply(self, count: int, min_layer: int = 2, mode: str = "redundant", seed: int = 0):
        """Bypass `count` layers, never touching the first `min_layer` layers or
        the final layer (they carry the most signal).

        mode:
          "redundant" (default) - bypass the `count` MOST redundant layers
                                   (highest cos(in,out)) per the calibrated ranking.
          "least"                - bypass the `count` LEAST redundant layers
                                   (control: should be much worse if the ranking
                                   carries real signal).
          "random"               - bypass `count` random eligible layers, seeded
                                   for reproducibility (control: distinguishes
                                   "our ranking picks bad layers" from "this model
                                   is brittle to ANY layer drop").
        """
        self.restore()
        if count <= 0 or self.ranking is None:
            return []
        protected = set(range(min_layer)) | {self.n - 1}
        eligible = [i for i in self.ranking if i not in protected]
        if mode == "redundant":
            chosen = eligible[:count]
        elif mode == "least":
            chosen = eligible[::-1][:count]
        elif mode == "random":
            import random
            rng = random.Random(seed)
            chosen = rng.sample(eligible, min(count, len(eligible)))
        else:
            raise ValueError(f"unknown mode: {mode}")
        for i in chosen:
            self.layers[i] = _BypassLayer(self.orig[i])
        self.pruned = sorted(chosen)
        self.mode = mode
        return self.pruned

    def restore(self):
        for i in range(self.n):
            self.layers[i] = self.orig[i]
        self.pruned = []

    def summary(self):
        return {
            "n_layers": self.n,
            "pruned": self.pruned,
            "ranking_top": self.ranking[:8] if self.ranking else None,
            "redundancy": [round(x, 4) for x in getattr(self, "_redundancy", [])],
        }

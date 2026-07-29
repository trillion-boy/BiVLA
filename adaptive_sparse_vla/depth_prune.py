"""Training-free decoder-layer bypass, shared by every LIBERO backbone.

Rank decoder layers by `1 - cos(layer_in, layer_out)` -- low means the layer
barely changes the representation, so it is redundant -- and replace the most
redundant with a pass-through. This attacks the autoregressive action decode,
which is ~70% of a control step (`docs/VISUAL_TOKENS_VS_LATENCY.md`) and pays
for every layer on every generated token, so unlike anything on the visual axis
it moves wall-clock.

Two modes:

- **static** (`prune=N`): bypass the same N layers all episode.
- **phase-adaptive** (`ctrl=True`): bypass `deep` during the precise
  approach+grasp, `shallow` once the policy has commanded the gripper closed
  for `close_steps` consecutive calls. One-way, so it cannot oscillate, and
  `deep` is a strict prefix of `shallow`, so the switch only ever adds layers.

**This lives in one place on purpose.** The cross-backbone claim is that
exploitable depth redundancy is a property of the backbone -- Emu3 absorbs 8
bypassed layers where Gemma2 broke at 1 -- and that only means something if
every backbone is measured and pruned by an identical rule. Two copies of this
logic would drift and silently invalidate the comparison.

What is *not* shared is how the per-layer redundancy is measured, because the
backbones expose different call paths: Emu3 can be run directly with
`output_hidden_states=True`, while OpenVLA's generate is wrapped inside
`predict_action`, so it is measured with forward hooks instead
(`measure_redundancy_with_hooks`). Both compute the same quantity on the same
thing -- layer input vs layer output on the real prompt's prefill.
"""
from __future__ import annotations

import math
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

try:
    from adaptive_sparse_vla.bypass_layer import BypassDecoderLayer
except Exception:  # when imported as a top-level package on sys.path
    from bypass_layer import BypassDecoderLayer


def find_decoder_layers(model) -> Optional[torch.nn.ModuleList]:
    """Locate the decoder stack across the wrappers the backbones use.

    Emu3 exposes it at `model.model.layers`; OpenVLA wraps a Llama inside
    Prismatic, so it sits at `model.language_model.model.layers`. Walking a
    candidate list beats hard-coding one path per backbone, which is how a
    silently-unpruned run happens.
    """
    candidates = (
        ("model", "layers"),
        ("language_model", "model", "layers"),
        ("language_model", "layers"),
        ("model", "language_model", "layers"),
        ("layers",),
    )
    for path in candidates:
        node = model
        for attr in path:
            node = getattr(node, attr, None)
            if node is None:
                break
        if isinstance(node, torch.nn.ModuleList) and len(node) > 0:
            return node
    return None


def measure_redundancy_with_hooks(
    layers: torch.nn.ModuleList, run_fn: Callable[[], object]
) -> Optional[List[float]]:
    """Per-layer `1 - cos(in, out)`, captured while `run_fn()` executes.

    Only the FIRST call of each layer is recorded. Generation calls every layer
    once per decoded token, and the prefill -- the first call -- is the one that
    sees the whole prompt; averaging in single-token decode steps would measure
    something else.

    Must run with the stack unpruned, or bypassed layers report ~0 redundancy
    (input == output) and rank themselves most-redundant forever.
    """
    scores: List[Optional[float]] = [None] * len(layers)
    handles = []

    def make_hook(idx: int):
        def hook(module, args, kwargs, output):
            if scores[idx] is not None:
                return
            inp = args[0] if args else kwargs.get("hidden_states")
            out = output[0] if isinstance(output, tuple) else output
            if inp is None or out is None or not torch.is_tensor(inp):
                return
            cos = torch.nn.functional.cosine_similarity(
                inp.float(), out.float(), dim=-1
            )
            scores[idx] = float(1.0 - cos.mean().item())
        return hook

    for i, layer in enumerate(layers):
        handles.append(layer.register_forward_hook(make_hook(i), with_kwargs=True))
    try:
        run_fn()
    finally:
        for h in handles:
            h.remove()
    if any(s is None for s in scores):
        return None
    return [float(s) for s in scores]


class DepthPruner:
    """Owns the bypass bookkeeping and the deep/shallow state machine."""

    def __init__(
        self,
        model,
        *,
        prune: int = 0,
        ctrl: bool = False,
        deep: int = 2,
        shallow: int = 8,
        close_steps: int = 2,
        min_layer: float = 0.5,
        min_gap: int = 1,
    ):
        self.model = model
        self.prune = max(0, int(prune))
        self.ctrl = bool(ctrl)
        self.deep = max(0, int(deep))
        self.shallow = max(0, int(shallow))
        self.close_steps = max(1, int(close_steps))
        self.min_layer = float(min_layer)
        self.min_gap = max(0, int(min_gap))

        self._originals: Dict[int, torch.nn.Module] = {}
        self._active: Tuple[int, ...] = ()
        self._ranking: List[int] = []
        self._ranking_ready = False
        self._state = "deep"
        self._calibrated = False
        self.close_gripper_num = 0
        self.episodes = 0
        self.switches = 0

    # -- enabled / layers ---------------------------------------------------
    @property
    def enabled(self) -> bool:
        return self.ctrl or self.prune > 0

    def layers(self) -> Optional[torch.nn.ModuleList]:
        return find_decoder_layers(self.model)

    def n_layers(self) -> int:
        layers = self.layers()
        return 0 if layers is None else len(layers)

    def announce(self, backbone: str = "") -> None:
        if not self.enabled:
            return
        n = self.n_layers()
        eligible = f"{int(self.min_layer * n)}..{n - 1}" if n else "?"
        tag = f"[depth{'/' + backbone if backbone else ''}]"
        if self.ctrl:
            print(f"{tag} phase-adaptive controller ON: deep={self.deep} "
                  f"shallow={self.shallow} switch after {self.close_steps} "
                  f"consecutive close-gripper calls; eligible layers {eligible} "
                  f"of {n}", flush=True)
        else:
            print(f"{tag} static pruning ON: bypass {self.prune} most-redundant "
                  f"layers; eligible {eligible} of {n}", flush=True)

    # -- bypass bookkeeping -------------------------------------------------
    def restore(self) -> None:
        layers = self.layers()
        if layers is None or not self._originals:
            return
        for idx, layer in self._originals.items():
            layers[idx] = layer
        self._originals = {}
        self._active = ()

    def apply(self, indices: Sequence[int]) -> None:
        layers = self.layers()
        if layers is None:
            return
        valid = tuple(
            i for i in sorted({int(x) for x in indices}) if 0 <= i < len(layers)
        )
        # Restore first: entering a new state from an old one must put the real
        # modules back, or bypasses accumulate and can never be undone.
        self.restore()
        if not valid:
            return
        cfg = getattr(self.model, "config", None)
        # Prismatic nests the LLM's own config; the outer config has no
        # attention shape, and a wrong head_dim makes a KV placeholder the
        # cache cannot concatenate.
        for attr in ("text_config", "llm_config", "language_model_config"):
            sub = getattr(cfg, attr, None)
            if sub is not None and getattr(sub, "num_attention_heads", None):
                cfg = sub
                break
        n_heads = getattr(cfg, "num_attention_heads", None)
        hidden = getattr(cfg, "hidden_size", None)
        num_kv_heads = getattr(cfg, "num_key_value_heads", None) or n_heads
        head_dim = getattr(cfg, "head_dim", None) or (
            (hidden // n_heads) if (hidden and n_heads) else None
        )
        for idx in valid:
            self._originals[idx] = layers[idx]
            layers[idx] = BypassDecoderLayer(
                idx, num_kv_heads=num_kv_heads, head_dim=head_dim
            )
        self._active = valid

    # -- ranking ------------------------------------------------------------
    def rank_layers(self, importance: Sequence[float]) -> List[int]:
        """Eligible layers, most-redundant first.

        Only the back `1 - min_layer` of the stack is eligible: early layers
        perform the large foundational transforms everything downstream depends
        on, and removing them corrupts the representation outright (measured on
        Gemma2: bypassing layers 2 and 4 made generation never terminate).

        Gap-respecting greedy first, so adjacent layers are not both bypassed
        while cheaper candidates remain, then the rest. Prefixes are nested, so
        `deep` is a strict prefix of `shallow`.
        """
        scores = np.asarray(importance, dtype=np.float32)
        n = int(scores.shape[0])
        start = int(math.floor(np.clip(self.min_layer, 0.0, 1.0) * n))
        candidates = list(range(start, n)) or list(range(n))
        ranked = sorted(candidates, key=lambda i: (float(scores[i]), i))
        ordered: List[int] = []
        for i in ranked:
            if any(abs(i - prev) <= self.min_gap for prev in ordered):
                continue
            ordered.append(i)
        for i in ranked:
            if i not in ordered:
                ordered.append(i)
        return ordered

    def needs_calibration(self) -> bool:
        if not self.enabled:
            return False
        return not (self._ranking_ready if self.ctrl else self._calibrated)

    def calibrate(self, importance: Optional[Sequence[float]]) -> None:
        if importance is None:
            return
        ranking = self.rank_layers(importance)
        if self.ctrl:
            self._ranking = ranking
            self._ranking_ready = True
            self._state = "deep"
            self.apply(ranking[:max(0, self.deep)])
            print(f"[depth] calibrated: ranking(top8)={ranking[:8]} state=deep "
                  f"bypass={list(self._active)}", flush=True)
        else:
            self._calibrated = True
            self.apply(ranking[:self.prune])
            print(f"[depth] calibrated: bypass={list(self._active)}", flush=True)

    # -- state machine ------------------------------------------------------
    def _apply_state(self) -> None:
        if not self._ranking:
            return
        count = self.deep if self._state == "deep" else self.shallow
        self.apply(self._ranking[:max(0, count)])

    def note_gripper(self, closed: bool) -> None:
        """Feed the policy's own commanded gripper; switch deep -> shallow once
        it has been closed for `close_steps` consecutive calls.

        The signal costs nothing (no env ground truth, no detector) and the
        switch is one-way, so the precise approach+grasp keeps full depth while
        transport and place run shallow.
        """
        if not self.ctrl:
            return
        self.close_gripper_num = self.close_gripper_num + 1 if closed else 0
        if (
            self._state == "deep"
            and self._ranking_ready
            and self.close_gripper_num >= self.close_steps
        ):
            self._state = "shallow"
            self._apply_state()
            self.switches += 1
            if self.switches == 1:
                print(f"      [depth] deep -> shallow (grasp confirmed): "
                      f"bypass={list(self._active)}", flush=True)

    def reset_episode(self) -> None:
        self.close_gripper_num = 0
        if not self.ctrl:
            return
        # Re-rank every episode, which needs the stack unpruned for the
        # measurement to describe the real model.
        self.episodes += 1
        self._state = "deep"
        self._ranking = []
        self._ranking_ready = False
        self.restore()

    # -- reporting ----------------------------------------------------------
    def summary(self) -> dict:
        out = {
            "depth_prune": self.prune,
            "depth_ctrl": self.ctrl,
            "bypassed_layers": list(self._active),
            "n_bypassed": len(self._active),
            "n_layers": self.n_layers(),
        }
        if self.ctrl:
            out.update({
                "depth_deep": self.deep,
                "depth_shallow": self.shallow,
                "depth_close_steps": self.close_steps,
                "depth_state_at_end": self._state,
                "episodes": self.episodes,
                "episodes_reaching_shallow": self.switches,
                # 0.0 means the controller never left deep, i.e. the run was
                # really --depth-prune <deep> under another name.
                "shallow_fraction": (
                    self.switches / self.episodes if self.episodes else None
                ),
            })
        return out

"""Calibration-set block-influence pruning for Llama-based OpenVLA.

This corrects three risks in the original notebook: it aggregates influence
over multiple prompts, protects the final decoder block, and enforces the
non-adjacency constraint instead of silently filling with adjacent blocks.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from typing import Any

import numpy as np
import torch


def find_decoder_layer_container(
    model: torch.nn.Module,
) -> tuple[torch.nn.Module, str, torch.nn.ModuleList] | None:
    candidates = (
        ("language_model", "model", "layers"),  # OpenVLA
        ("llm_backbone", "llm", "model", "layers"),  # Prismatic MiniVLA
        ("llm_backbone", "llm", "layers"),
        ("model", "language_model", "model", "layers"),
        ("model", "layers"),
        ("language_model", "layers"),
        ("layers",),
    )
    for path in candidates:
        node: Any = model
        for attribute in path[:-1]:
            node = getattr(node, attribute, None)
            if node is None:
                break
        if node is None:
            continue
        layers = getattr(node, path[-1], None)
        if isinstance(layers, torch.nn.ModuleList) and layers:
            return node, path[-1], layers
    return None


def find_decoder_layers(model: torch.nn.Module) -> torch.nn.ModuleList | None:
    found = find_decoder_layer_container(model)
    return found[2] if found is not None else None


def collect_block_influence(
    layers: Sequence[torch.nn.Module], run_fns: Iterable[Callable[[], Any]]
) -> list[float]:
    """Average `1-cos(input, output)` over a calibration set's prefills."""
    sums = np.zeros(len(layers), dtype=np.float64)
    counts = np.zeros(len(layers), dtype=np.int64)
    seen_in_run: set[int] = set()
    handles = []

    def make_hook(index: int):
        def hook(module, args, kwargs, output):
            if index in seen_in_run:
                return
            hidden_in = args[0] if args else kwargs.get("hidden_states")
            hidden_out = output[0] if isinstance(output, tuple) else output
            if not (torch.is_tensor(hidden_in) and torch.is_tensor(hidden_out)):
                return
            cosine = torch.nn.functional.cosine_similarity(
                hidden_in.detach().float(), hidden_out.detach().float(), dim=-1
            )
            sums[index] += 1.0 - float(cosine.mean().item())
            counts[index] += 1
            seen_in_run.add(index)

        return hook

    for index, layer in enumerate(layers):
        handles.append(layer.register_forward_hook(make_hook(index), with_kwargs=True))
    try:
        for run_fn in run_fns:
            seen_in_run.clear()
            run_fn()
    finally:
        for handle in handles:
            handle.remove()

    missing = np.flatnonzero(counts == 0).tolist()
    if missing:
        raise RuntimeError(f"decoder hooks did not observe layers {missing}")
    return (sums / counts).astype(float).tolist()


def select_non_adjacent(
    influence: Sequence[float],
    count: int,
    *,
    min_layer_fraction: float = 0.25,
    protect_last: int = 1,
    min_gap: int = 1,
) -> tuple[int, ...]:
    """Select low-influence blocks while strictly enforcing all safeguards."""
    scores = np.asarray(influence, dtype=np.float64)
    if scores.ndim != 1 or scores.size == 0:
        raise ValueError("influence must be a non-empty one-dimensional sequence")
    if count < 0:
        raise ValueError("count must be non-negative")
    start = int(np.ceil(np.clip(min_layer_fraction, 0.0, 1.0) * len(scores)))
    stop = max(start, len(scores) - int(protect_last))
    ranked = sorted(range(start, stop), key=lambda i: (scores[i], i))
    chosen: list[int] = []
    for index in ranked:
        if all(abs(index - prior) > min_gap for prior in chosen):
            chosen.append(index)
            if len(chosen) == count:
                return tuple(sorted(chosen))
    if count:
        raise ValueError(
            f"cannot select {count} protected, gap-constrained layers; "
            f"maximum found was {len(chosen)}"
        )
    return ()


class BypassDecoderLayer(torch.nn.Module):
    """Identity block that keeps a DynamicCache's layer indices contiguous."""

    def __init__(self, layer_idx: int, num_kv_heads: int, head_dim: int):
        super().__init__()
        self.layer_idx = int(layer_idx)
        self.num_kv_heads = int(num_kv_heads)
        self.head_dim = int(head_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        *args: Any,
        past_key_value: Any = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs: Any,
    ) -> tuple[Any, ...]:
        if use_cache and past_key_value is not None and hasattr(past_key_value, "update"):
            batch, sequence = hidden_states.shape[:2]
            placeholder = torch.zeros(
                batch,
                self.num_kv_heads,
                sequence,
                self.head_dim,
                dtype=hidden_states.dtype,
                device=hidden_states.device,
            )
            past_key_value.update(placeholder, placeholder, self.layer_idx, {})
        outputs: tuple[Any, ...] = (hidden_states,)
        if output_attentions:
            outputs += (None,)
        if use_cache:
            outputs += (past_key_value,)
        return outputs


class StaticDepthPruner:
    """Structurally remove selected blocks, reindex cache slots, and restore.

    True removal avoids the zero-KV allocation overhead of an identity bypass
    and matches the operation evaluated by ShortGPT/EfficientVLA.
    """

    def __init__(self, model: torch.nn.Module):
        self.model = model
        self._original_layers: list[torch.nn.Module] | None = None
        self._original_layer_indices: dict[int, int] = {}
        self._original_config_depths: list[tuple[Any, int]] = []
        self.active: tuple[int, ...] = ()

    @property
    def layers(self) -> torch.nn.ModuleList:
        layers = find_decoder_layers(self.model)
        if layers is None:
            raise RuntimeError("could not locate the decoder ModuleList")
        return layers

    def apply(self, indices: Sequence[int]) -> None:
        self.restore()
        valid = tuple(sorted(set(map(int, indices))))
        if any(index < 0 or index >= len(self.layers) for index in valid):
            raise IndexError(f"invalid decoder indices: {valid}")

        found = find_decoder_layer_container(self.model)
        if found is None:
            raise RuntimeError("could not locate the decoder ModuleList")
        parent, attribute, layers = found
        self._original_layers = list(layers)
        removed = set(valid)
        retained = [layer for index, layer in enumerate(layers) if index not in removed]

        for new_index, layer in enumerate(retained):
            attention = getattr(layer, "self_attn", None)
            if attention is not None and hasattr(attention, "layer_idx"):
                self._original_layer_indices[id(attention)] = int(attention.layer_idx)
                attention.layer_idx = new_index
        setattr(parent, attribute, torch.nn.ModuleList(retained))

        configs = [
            getattr(self.model, "config", None),
            getattr(getattr(self.model, "language_model", None), "config", None),
            getattr(getattr(self.model, "config", None), "text_config", None),
        ]
        seen_configs: set[int] = set()
        for config in configs:
            if config is not None and hasattr(config, "num_hidden_layers"):
                if id(config) in seen_configs:
                    continue
                seen_configs.add(id(config))
                self._original_config_depths.append((config, int(config.num_hidden_layers)))
                config.num_hidden_layers = len(retained)
        self.active = valid

    def restore(self) -> None:
        if self._original_layers is None:
            self.active = ()
            return
        found = find_decoder_layer_container(self.model)
        if found is None:
            raise RuntimeError("could not locate the pruned decoder ModuleList")
        parent, attribute, _ = found
        for layer in self._original_layers:
            attention = getattr(layer, "self_attn", None)
            if attention is not None and id(attention) in self._original_layer_indices:
                attention.layer_idx = self._original_layer_indices[id(attention)]
        setattr(parent, attribute, torch.nn.ModuleList(self._original_layers))
        for config, depth in self._original_config_depths:
            config.num_hidden_layers = depth
        self._original_layers = None
        self._original_layer_indices.clear()
        self._original_config_depths.clear()
        self.active = ()

    def __enter__(self) -> "StaticDepthPruner":
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.restore()

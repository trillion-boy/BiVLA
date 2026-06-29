"""
Training-free Token Merging (ToMe) for SpatialVLA's frozen SigLIP ViT.

Goal (project direction): cut *visual-token* compute inside the ViT so the policy
runs faster, WITHOUT training and WITHOUT moving the frozen backbone out of
distribution.

How this stays OOD-safe
-----------------------
* We merge tokens *between* SigLIP encoder layers (ToMe, ICLR'23), which the
  original method shows works on off-the-shelf frozen ViTs.
* Merging is a *weighted average* of similar tokens (NOT a hard drop) — redundant
  background patches collapse together; distinctive (important) patches have no
  similar neighbour and survive on their own. This is exactly "merge the
  background, keep the important region sharp".
* After the last merged layer we **unmerge** (broadcast each merged cluster back
  to all of its original patch positions), so the tensor handed to the projector
  / Gemma2 has the **same token count and grid layout** as the baseline. The
  language model therefore sees nothing out of distribution — the only change is
  that a group of merged background patches now carry an identical (averaged)
  feature. Latency is saved because the encoder's middle layers run on fewer
  tokens; correctness is preserved because the layout downstream is untouched.

Optional protection
-------------------
A caller can supply a per-patch "importance" map (e.g. AutoGaze saliency, or a
cheap centre prior). Protected patches are never merged away and never absorb a
merge, so important regions keep full resolution. With no map supplied, pure
similarity-based ToMe already preserves distinctive regions.

This module is self-contained (only torch) and patches a SiglipVisionModel in
place; call `remove_tome(vision_tower)` to restore the original forward.
"""

from __future__ import annotations

import types
from typing import Callable, Optional

import torch


def _vision_output(last_hidden_state):
    """Wrap features in a SigLIP-style output object.

    Uses transformers' BaseModelOutputWithPooling when available (the real
    SpatialVLA runtime), else a tiny stand-in exposing `.last_hidden_state`
    so the module is importable/testable without transformers installed.
    """
    try:
        from transformers.modeling_outputs import BaseModelOutputWithPooling
        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=None,
            hidden_states=None,
            attentions=None,
        )
    except Exception:
        class _Out:
            pass
        o = _Out()
        o.last_hidden_state = last_hidden_state
        o.pooler_output = None
        return o


# --------------------------------------------------------------------------- #
# Core ToMe primitive: bipartite soft matching                                 #
# --------------------------------------------------------------------------- #
def bipartite_soft_matching(
    metric: torch.Tensor,
    r: int,
    protected: Optional[torch.Tensor] = None,
):
    """Return a `merge` callable that fuses `r` tokens (ToMe, ICLR'23).

    Args:
        metric:    (B, N, C) token features used to measure similarity.
        r:         number of tokens to remove this round.
        protected: (B, N) bool — True for tokens that must stay untouched
                   (never merged away, never absorb a merge).

    Returns:
        merge(x, mode="mean") -> (B, N - r, C)
    """
    B, N, _ = metric.shape
    r = min(r, (N - 2) // 2)  # need at least 1 token left in each bipartite set
    if r <= 0:
        return None

    with torch.no_grad():
        m = metric / (metric.norm(dim=-1, keepdim=True) + 1e-6)
        # Even tokens form set A (merge sources), odd tokens form set B (sinks).
        a, b = m[:, ::2, :], m[:, 1::2, :]
        scores = a @ b.transpose(-1, -2)          # (B, |A|, |B|)

        if protected is not None:
            pa = protected[:, ::2]                # (B, |A|)
            pb = protected[:, 1::2]               # (B, |B|)
            # A protected A-token must never be merged away → kill its rows.
            scores = scores.masked_fill(pa[..., None], -torch.inf)
            # A protected B-token must never absorb a merge → kill its columns.
            scores = scores.masked_fill(pb[:, None, :], -torch.inf)

        node_max, node_idx = scores.max(dim=-1)   # best B-partner for each A-token
        edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]

        unm_idx = edge_idx[:, r:, :]              # A-tokens kept separate
        src_idx = edge_idx[:, :r, :]              # A-tokens merged into B
        dst_idx = node_idx[..., None].gather(dim=-2, index=src_idx)

    def merge(x: torch.Tensor, mode: str = "mean") -> torch.Tensor:
        src, dst = x[:, ::2, :], x[:, 1::2, :]
        n, _, c = src.shape
        unm = src.gather(dim=-2, index=unm_idx.expand(n, unm_idx.shape[1], c))
        src = src.gather(dim=-2, index=src_idx.expand(n, r, c))
        dst = dst.scatter_reduce(-2, dst_idx.expand(n, r, c), src, reduce=mode)
        return torch.cat([unm, dst], dim=1)

    return merge


def _merge_source(merge: Callable, source: torch.Tensor) -> torch.Tensor:
    """Track which *original* patches each current token represents.

    `source` is (B, N_cur, N_orig); merging with mode="amax" keeps the union of
    contributing original indices marked. After all merges, argmax over the
    N_cur axis tells us, for every original patch, which surviving token holds
    its (merged) feature — used to unmerge back to the full grid.
    """
    return merge(source, mode="amax")


# --------------------------------------------------------------------------- #
# Patching a SigLIP vision tower                                               #
# --------------------------------------------------------------------------- #
def _locate_vision_model(vision_tower):
    """SiglipVisionModel -> SiglipVisionTransformer (.vision_model)."""
    vm = getattr(vision_tower, "vision_model", None)
    if vm is None or not hasattr(vm, "encoder") or not hasattr(vm, "embeddings"):
        raise AttributeError(
            "ToMe expects a SiglipVisionModel with .vision_model.{embeddings,encoder,"
            "post_layernorm}; got " + type(vision_tower).__name__
        )
    return vm


def apply_tome_to_siglip(
    vision_tower,
    r: int = 8,
    num_merge_layers: int = 6,
    protect_provider: Optional[Callable[[int, int, torch.device], Optional[torch.Tensor]]] = None,
    verbose: bool = True,
):
    """Patch a frozen SigLIP vision tower to merge `r` tokens in each of the
    first `num_merge_layers` encoder layers, then unmerge back to full length.

    Args:
        vision_tower:    SiglipVisionModel (e.g. SpatialVLA's self.model.vision_tower).
        r:               tokens merged per layer (more = faster, more approximate).
        num_merge_layers: how many *early* encoder layers merge in. Early layers
                          carry the most redundancy and the longest compute tail.
        protect_provider: optional fn (B, N, device) -> (B, N) bool importance
                          mask; True = keep at full resolution.
        verbose:         print a one-line summary on patch.
    """
    vm = _locate_vision_model(vision_tower)
    if getattr(vision_tower, "_tome_patched", False):
        remove_tome(vision_tower)

    vision_tower._tome_orig_forward = vision_tower.forward
    layers = vm.encoder.layers
    n_layers = len(layers)
    merge_layers = set(range(min(num_merge_layers, n_layers)))

    def tome_forward(self, pixel_values, *args, **kwargs):
        # 1. patch embeddings + position embeddings (position baked in here)
        hidden = vm.embeddings(pixel_values)
        B, N0, C = hidden.shape

        protect = None
        if protect_provider is not None:
            protect = protect_provider(B, N0, hidden.device)
            if protect is not None:
                protect = protect.to(hidden.device).bool()

        # source tracks original-patch membership for end-of-stack unmerge
        source = torch.eye(N0, device=hidden.device, dtype=hidden.dtype)
        source = source[None].expand(B, N0, N0).clone()
        cur_protect = protect

        # 2. encoder layers, merging in the early ones
        for li, layer in enumerate(layers):
            layer_out = layer(hidden, attention_mask=None, output_attentions=False)
            hidden = layer_out[0] if isinstance(layer_out, tuple) else layer_out
            if li in merge_layers:
                merge = bipartite_soft_matching(hidden, r, protected=cur_protect)
                if merge is not None:
                    hidden = merge(hidden, mode="mean")
                    source = merge(source, mode="amax")
                    if cur_protect is not None:
                        # carry protection forward: a merged token stays protected
                        # if any of its members were protected
                        pf = cur_protect.to(hidden.dtype)[..., None]
                        cur_protect = (merge(pf, mode="amax")[..., 0] > 0.5)

        hidden = vm.post_layernorm(hidden)

        # 3. unmerge: each original patch reads its surviving token's feature
        #    -> (B, N0, C); identical layout to baseline => no downstream OOD
        owner = source.argmax(dim=1)                      # (B, N0) surviving-token idx
        restored = hidden.gather(1, owner[..., None].expand(B, N0, C))
        return _vision_output(restored)

    vision_tower.forward = types.MethodType(tome_forward, vision_tower)
    vision_tower._tome_patched = True
    if verbose:
        kept = "with importance protection" if protect_provider else "pure similarity"
        print(
            f"[ToMe] patched SigLIP: r={r}/layer x {len(merge_layers)} early layers "
            f"({kept}); tokens restored to full grid at output.",
            flush=True,
        )
    return vision_tower


def remove_tome(vision_tower):
    """Restore the original (un-merged) vision-tower forward."""
    if getattr(vision_tower, "_tome_patched", False):
        vision_tower.forward = vision_tower._tome_orig_forward
        del vision_tower._tome_orig_forward
        vision_tower._tome_patched = False
    return vision_tower


# --------------------------------------------------------------------------- #
# Built-in protection priors (training-free, zero added latency)               #
# --------------------------------------------------------------------------- #
def center_protect_provider(keep_ratio: float = 0.25):
    """Protect a centred square covering ~`keep_ratio` of patches.

    A cheap, model-free saliency prior: WidowX-Bridge keeps the manipulated
    object roughly centred. No extra network, no latency.
    """
    def provider(B: int, N: int, device: torch.device) -> Optional[torch.Tensor]:
        g = int(round(N ** 0.5))
        if g * g != N:
            return None
        half = max(1, int(round(g * (keep_ratio ** 0.5) / 2)))
        lo, hi = g // 2 - half, g // 2 + half
        grid = torch.zeros(g, g, dtype=torch.bool, device=device)
        grid[lo:hi, lo:hi] = True
        return grid.view(1, N).expand(B, N).contiguous()

    return provider

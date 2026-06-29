"""
Training-free FastV-style visual-token pruning for UniVLA's frozen Emu3 LLM.

Why this (and not ToMe) for UniVLA
----------------------------------
UniVLA/Emu3 has *no* ViT: images become a fixed rectangular grid of **discrete
VQ tokens** declared in the text prefix, then the Emu3 LLM reads them. You cannot
average two discrete IDs and dropping any breaks the declared grid, so reducing
tokens at the *input* is structurally out-of-distribution. The latency instead
lives in the **LLM** chewing through every visual token at every layer and every
decode step.

FastV (ECCV'24) cuts exactly that, training-free and OOD-tolerantly:
  * Let the first `k` layers run on the full grid so the model aggregates the
    scene (early layers carry the spatial structure).
  * At layer `k`, read the model's **own attention** to score visual tokens, keep
    the most-attended fraction, and **drop the rest from the hidden stream** for
    all layers > k. The discrete input grid is untouched (no input OOD); pruning
    happens in continuous latent space after the scene is understood.
  * The dropped tokens never enter the KV cache of the deep layers, so the (many)
    autoregressive action-decode steps attend to a much shorter cache => real
    wall-clock savings.

"Foveation feel": the highly-attended (task-relevant) tokens keep full attention
through every layer; the periphery is dropped after layer k. No external module
(no GroundingDINO) — the importance signal is the LLM's own attention.

This module monkeypatches `Emu3Model.forward` (the decoder stack). Because UniVLA
ships this exact Emu3 code via trust_remote_code, we reuse the *loaded* model's
own rotary / repeat_kv so the importance attention matches the model bit-for-bit.
Pruning is applied ONLY on the generation prefill (q_len > 1 and use_cache);
calibration / no-cache forwards pass through unchanged.
"""

from __future__ import annotations

import math
import types
from typing import Optional

import torch


# --------------------------------------------------------------------------- #
# Importance: attention the generation cursor pays to each visual token        #
# --------------------------------------------------------------------------- #
def _rope_fns_for(attn_module):
    """Import apply_rotary_pos_emb / repeat_kv from the *loaded* model's module
    so RoPE/GQA match the running Emu3 exactly. Returns (apply_rope, repeat_kv)
    or (None, None) if unavailable (caller falls back to a no-RoPE proxy)."""
    try:
        import importlib
        mod = importlib.import_module(type(attn_module).__module__)
        return getattr(mod, "apply_rotary_pos_emb"), getattr(mod, "repeat_kv")
    except Exception:
        return None, None


def _visual_importance(layer, hidden, position_ids, visual_mask):
    """Score each token by the attention the LAST query position gives it at
    this layer (FastV's signal), averaged over heads. Falls back to a RoPE-free
    cosine proxy if the rotary path is unavailable. Returns (B, N) float."""
    attn = layer.self_attn
    B, N, _ = hidden.shape
    hn = layer.input_layernorm(hidden)
    apply_rope, repeat_kv = _rope_fns_for(attn)

    try:
        if apply_rope is None:
            raise RuntimeError("no rope fns")
        nh, nkv, hd = attn.num_heads, attn.num_key_value_heads, attn.head_dim
        q = attn.q_proj(hn).view(B, N, nh, hd).transpose(1, 2)
        k = attn.k_proj(hn).view(B, N, nkv, hd).transpose(1, 2)
        v = attn.v_proj(hn).view(B, N, nkv, hd).transpose(1, 2)
        cos, sin = attn.rotary_emb(v, seq_len=N)
        q, k = apply_rope(q, k, cos, sin, position_ids)
        k = repeat_kv(k, attn.num_key_value_groups)
        # attention from the last (cursor) row over all keys
        q_last = q[:, :, -1:, :]                                  # (B, nh, 1, hd)
        scores = (q_last @ k.transpose(2, 3)) / math.sqrt(hd)      # (B, nh, 1, N)
        attnw = torch.softmax(scores.float(), dim=-1)
        imp = attnw[:, :, 0, :].mean(dim=1)                        # (B, N)
    except Exception:
        # robust fallback: cosine similarity to the last token's hidden state
        ref = hn[:, -1:, :]
        imp = torch.nn.functional.cosine_similarity(hn, ref, dim=-1).clamp_min(0)

    # never let a non-visual token be pruned: give them +inf importance
    imp = imp.masked_fill(~visual_mask, float("inf"))
    return imp


def _build_keep_index(importance, visual_mask, keep_ratio):
    """Keep all non-visual tokens + the top `keep_ratio` of visual tokens.
    Returns a sorted LongTensor of kept positions (assumes batch size 1, which
    is how UniVLA steps the policy)."""
    B, N = importance.shape
    assert B == 1, "FastV pruning path assumes batch size 1 (UniVLA policy step)"
    n_vis = int(visual_mask[0].sum().item())
    keep_vis = max(1, int(round(n_vis * float(keep_ratio))))
    # +inf for non-visual => they always rank first and are always kept
    n_keep = (N - n_vis) + keep_vis
    order = torch.argsort(importance[0], descending=True)
    keep = order[:n_keep]
    return torch.sort(keep).values


# --------------------------------------------------------------------------- #
# Causal mask helpers (we manage masks ourselves once tokens are pruned)       #
# --------------------------------------------------------------------------- #
def _causal_mask(q_len, kv_len, dtype, device):
    """Additive (float) causal mask of shape (1, 1, q_len, kv_len)."""
    min_val = torch.finfo(dtype).min
    m = torch.full((q_len, kv_len), min_val, dtype=dtype, device=device)
    # allow attending to keys at or before each query (keys aligned to the right)
    offset = kv_len - q_len
    idx = torch.arange(q_len, device=device)
    for_keys = torch.arange(kv_len, device=device)
    allow = for_keys[None, :] <= (idx[:, None] + offset)
    m = m.masked_fill(allow, 0.0)
    return m[None, None]


# --------------------------------------------------------------------------- #
# The FastV-aware Emu3Model.forward                                            #
# --------------------------------------------------------------------------- #
def apply_fastv(model, k_layer: int = 3, keep_ratio: float = 0.4, verbose: bool = True):
    """Patch an Emu3 causal LM so its decoder stack prunes visual tokens after
    layer `k_layer`, keeping `keep_ratio` of them.

    Args:
        model:      Emu3ForCausalLM (the policy's self.model). model.model is the
                    Emu3Model whose forward we replace.
        k_layer:    prune after this many full layers (FastV's K). 2-4 typical.
        keep_ratio: fraction of visual tokens kept past layer K (smaller=faster).

    The caller sets `model._fastv_visual_mask` to a (B, N) bool tensor (True =
    visual/prunable) right before each generation. Pruning fires only on the
    prefill (q_len > 1 with use_cache); decode/calibration pass through.
    """
    base = model.model  # Emu3Model
    if getattr(base, "_fastv_patched", False):
        remove_fastv(model)

    base._fastv_orig_forward = base.forward
    base._fastv_k = int(k_layer)
    base._fastv_keep_ratio = float(keep_ratio)
    model._fastv_visual_mask = None
    base._fastv_last_kept = None

    from transformers.modeling_outputs import BaseModelOutputWithPast
    try:
        from transformers.cache_utils import DynamicCache
    except Exception:
        DynamicCache = None

    def fastv_forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        B, S, _ = inputs_embeds.shape
        device, dtype = inputs_embeds.device, inputs_embeds.dtype
        K = base._fastv_k

        # Calibration / training / no-cache forwards (e.g. layer-pruning calibration)
        # run the original stack untouched.
        if not use_cache:
            return base._fastv_orig_forward(
                inputs_embeds=inputs_embeds, attention_mask=attention_mask,
                position_ids=position_ids, past_key_values=past_key_values,
                use_cache=use_cache, output_attentions=output_attentions,
                output_hidden_states=output_hidden_states, return_dict=return_dict,
            )

        if past_key_values is None and DynamicCache is not None:
            past_key_values = DynamicCache()
        hidden = self.dropout(inputs_embeds)

        # ============================ DECODE (q_len == 1) ====================
        # Every layer attends to all of its OWN cached keys (lengths differ between
        # the unpruned early layers and the pruned deep layers), so we pass mask=None
        # and a per-layer position equal to that layer's current cache length.
        if S == 1:
            for li, layer in enumerate(self.layers):
                try:
                    cur = past_key_values.get_seq_length(li)
                except Exception:
                    cur = 0
                pos = torch.tensor([[cur]], device=device, dtype=torch.long)
                hidden = layer(
                    hidden, attention_mask=None, position_ids=pos,
                    past_key_value=past_key_values, output_attentions=False,
                    use_cache=True,
                )[0]
            hidden = self.norm(hidden)
            return BaseModelOutputWithPast(
                last_hidden_state=hidden, past_key_values=past_key_values)

        # ============================ PREFILL (q_len > 1) ====================
        vmask = getattr(model, "_fastv_visual_mask", None)
        prune = vmask is not None and vmask.shape[1] == S
        if vmask is not None:
            vmask = vmask.to(device)
        full_pos = torch.arange(S, device=device)[None]
        kept_idx = None

        for li, layer in enumerate(self.layers):
            if prune and li == K:
                # FastV: score visual tokens by this layer's attention, drop the
                # weak ones, and RE-INDEX the survivors to contiguous positions
                # (Emu3's rotary is computed for exactly seq_len rows, so kept
                # tokens must use 0..M-1 to stay in range).
                imp = _visual_importance(layer, hidden, full_pos, vmask)
                kept_idx = _build_keep_index(imp, vmask, base._fastv_keep_ratio)
                hidden = hidden.index_select(1, kept_idx)
                base._fastv_last_kept = kept_idx

            m = hidden.shape[1]
            mask = _causal_mask(m, m, dtype, device)
            pos = full_pos if kept_idx is None else torch.arange(m, device=device)[None]
            hidden = layer(
                hidden, attention_mask=mask, position_ids=pos,
                past_key_value=past_key_values, output_attentions=False,
                use_cache=True,
            )[0]

        hidden = self.norm(hidden)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden, past_key_values=past_key_values)

    base.forward = types.MethodType(fastv_forward, base)
    base._fastv_patched = True
    if verbose:
        print(
            f"[FastV] patched Emu3 LLM: prune visual tokens after layer {k_layer}, "
            f"keep {keep_ratio:.0%} (training-free, attention-guided, input grid intact).",
            flush=True,
        )
    return model


def remove_fastv(model):
    base = getattr(model, "model", model)
    if getattr(base, "_fastv_patched", False):
        base.forward = base._fastv_orig_forward
        del base._fastv_orig_forward
        base._fastv_patched = False
    if hasattr(model, "_fastv_visual_mask"):
        model._fastv_visual_mask = None
    return model


# --------------------------------------------------------------------------- #
# Locating the prunable visual-token span in an Emu3 prompt                    #
# --------------------------------------------------------------------------- #
def visual_mask_from_input_ids(input_ids, tokenizer):
    """Mark visual (prunable) positions: the run of `<|visual token N|>` ids that
    sits between Emu3's img_token marker and the eof/eoi terminators. Returns a
    (B, N) bool tensor; if the markers can't be found, returns all-False (=> no
    pruning, safe)."""
    def tid(name):
        t = getattr(tokenizer, name, None)
        if t is None:
            return None
        return tokenizer.convert_tokens_to_ids(t) if isinstance(t, str) else t

    img_id = tid("img_token")
    eof_id = tid("eof_token")
    eoi_id = tid("eoi_token")
    boi_id = tid("boi_token")

    ids = input_ids if input_ids.ndim == 2 else input_ids[None]
    B, N = ids.shape
    mask = torch.zeros(B, N, dtype=torch.bool, device=ids.device)
    for b in range(B):
        row = ids[b].tolist()
        inside = False
        for j, t in enumerate(row):
            if img_id is not None and t == img_id:
                inside = True
                continue
            if t in (eof_id, eoi_id, boi_id):
                inside = False
                continue
            if inside:
                mask[b, j] = True
    return mask

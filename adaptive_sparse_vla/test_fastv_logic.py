"""CPU unit test for FastV pruning bookkeeping (no real Emu3, no GPU).

Validates the invariants we rely on:
  1. visual_mask_from_input_ids marks exactly the tokens between img_token and
     eof/eoi as prunable;
  2. _build_keep_index keeps ALL non-visual tokens + top keep_ratio of visual
     tokens, and always keeps the last (cursor) token;
  3. the patched Emu3Model.forward prunes the visual tokens on a prefill
     (q_len>1) -> shorter sequence, non-visual tokens preserved, last token kept;
  4. a decode-style call (q_len==1) / no-mask passes through unchanged.

The mock layers fall back to the RoPE-free importance proxy (the real RoPE path
only exists in the loaded Emu3 module), which is exactly what we want to test the
pruning mechanics independent of the attention maths.
"""
import torch
import torch.nn as nn

from fastv_emu3 import (
    apply_fastv,
    _build_keep_index,
    visual_mask_from_input_ids,
)


# --------------------------------------------------------------------------- #
def test_build_keep_index():
    N = 20
    vmask = torch.zeros(1, N, dtype=torch.bool)
    vmask[0, 2:18] = True                     # 16 visual tokens, 4 non-visual
    imp = torch.zeros(1, N)
    imp[0] = torch.arange(N).float()          # higher index = more important
    imp = imp.masked_fill(~vmask, float("inf"))
    keep = _build_keep_index(imp, vmask, keep_ratio=0.25)  # keep 4 of 16 visual
    keep_set = set(keep.tolist())
    # all 4 non-visual kept
    nonvis = [0, 1, 18, 19]
    assert all(i in keep_set for i in nonvis), keep_set
    # exactly 4 + 4 = 8 kept
    assert len(keep_set) == (N - 16) + 4, len(keep_set)
    # last token (cursor) kept
    assert (N - 1) in keep_set
    print(f"[ok] keep_index: {len(keep_set)} kept ({nonvis} + top-4 visual)")


def test_visual_mask_span():
    class Tok:
        img_token, eof_token, eoi_token, boi_token = 100, 101, 102, 103
        def convert_tokens_to_ids(self, t):
            return t
    tok = Tok()
    # [boi, size, img, v,v,v,v, eof, eoi, text, text]
    ids = torch.tensor([[103, 7, 100, 200, 201, 202, 203, 101, 102, 5, 6]])
    mask = visual_mask_from_input_ids(ids, tok)
    assert mask[0].tolist() == [False, False, False, True, True, True, True,
                                False, False, False, False], mask[0].tolist()
    print(f"[ok] visual span: {int(mask.sum())} visual tokens detected")


# --------------------------------------------------------------------------- #
class _MockAttn(nn.Module):
    pass


class _MockLayer(nn.Module):
    def __init__(self, C, layer_idx):
        super().__init__()
        self.input_layernorm = nn.LayerNorm(C)
        self.self_attn = _MockAttn()
        self.layer_idx = layer_idx

    def forward(self, hidden, attention_mask=None, position_ids=None,
                past_key_value=None, output_attentions=False, use_cache=False):
        L = hidden.shape[1]
        # identity layer; assert mask width matches current seq length
        if attention_mask is not None:
            assert attention_mask.shape[-1] == L, (attention_mask.shape, hidden.shape)
        # realistic KV-cache update so per-layer lengths accumulate like Emu3
        if use_cache and past_key_value is not None:
            k = torch.zeros(1, 1, L, 1)
            past_key_value.update(k, k.clone(), self.layer_idx, {})
        out = (hidden,)
        if use_cache:
            out = out + (past_key_value,)
        return out


class _Cfg:
    use_cache = True
    use_return_dict = True


class _MockEmu3Model(nn.Module):
    def __init__(self, V=300, C=16, n_layers=8):
        super().__init__()
        self.config = _Cfg()
        self.embed_tokens = nn.Embedding(V, C)
        self.dropout = nn.Dropout(0.0)
        self.norm = nn.LayerNorm(C)
        self.layers = nn.ModuleList([_MockLayer(C, i) for i in range(n_layers)])

    def forward(self, input_ids=None, attention_mask=None, position_ids=None,
                past_key_values=None, inputs_embeds=None, use_cache=None,
                output_attentions=None, output_hidden_states=None, return_dict=None):
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        h = self.dropout(inputs_embeds)
        for layer in self.layers:
            h = layer(h, attention_mask=None, position_ids=position_ids,
                      use_cache=bool(use_cache))[0]
        h = self.norm(h)
        from transformers.modeling_outputs import BaseModelOutputWithPast
        return BaseModelOutputWithPast(last_hidden_state=h, past_key_values=past_key_values)


class _MockEmu3ForCausalLM(nn.Module):
    def __init__(self, **kw):
        super().__init__()
        self.model = _MockEmu3Model(**kw)


def test_patched_prefill_prunes():
    torch.manual_seed(0)
    N, C = 40, 16
    lm = _MockEmu3ForCausalLM(C=C, n_layers=8)
    apply_fastv(lm, k_layer=3, keep_ratio=0.5, verbose=True)

    # 30 visual tokens (idx 2..32), rest non-visual; last token is the cursor
    vmask = torch.zeros(1, N, dtype=torch.bool)
    vmask[0, 2:32] = True
    lm._fastv_visual_mask = vmask

    ids = torch.randint(0, 300, (1, N))
    from transformers.cache_utils import DynamicCache
    out = lm.model.forward(input_ids=ids, use_cache=True,
                           past_key_values=DynamicCache(), return_dict=True)
    kept = lm.model._fastv_last_kept
    n_vis = int(vmask.sum())
    expected = (N - n_vis) + max(1, round(n_vis * 0.5))
    assert out.last_hidden_state.shape[1] == expected, (
        out.last_hidden_state.shape, expected)
    # every non-visual position survived
    keep_set = set(kept.tolist())
    nonvis = [i for i in range(N) if not vmask[0, i]]
    assert all(i in keep_set for i in nonvis)
    assert (N - 1) in keep_set                       # cursor kept
    print(f"[ok] prefill pruned {N} -> {out.last_hidden_state.shape[1]} "
          f"(kept all {len(nonvis)} non-visual + {expected-len(nonvis)} visual)")


def test_decode_passthrough():
    N, C = 40, 16
    lm = _MockEmu3ForCausalLM(C=C, n_layers=8)
    apply_fastv(lm, k_layer=3, keep_ratio=0.5, verbose=False)
    lm._fastv_visual_mask = torch.zeros(1, N, dtype=torch.bool)
    lm._fastv_visual_mask[0, 2:32] = True
    # decode step: single new token, no pruning should happen
    ids = torch.randint(0, 300, (1, 1))
    from transformers.cache_utils import DynamicCache
    out = lm.model.forward(input_ids=ids, use_cache=True,
                           past_key_values=DynamicCache(), return_dict=True)
    assert out.last_hidden_state.shape[1] == 1, out.last_hidden_state.shape
    print("[ok] decode (q_len==1) passes through unchanged")


def test_prefill_then_decode_heterogeneous_cache():
    """The real-world flow: pruned prefill builds a cache whose early layers are
    full (N) and deep layers are short (M); then several decode steps run without
    shape errors and each layer's cache grows from its own length."""
    torch.manual_seed(2)
    N, C, K, nL = 40, 16, 3, 8
    lm = _MockEmu3ForCausalLM(C=C, n_layers=nL)
    apply_fastv(lm, k_layer=K, keep_ratio=0.5, verbose=False)
    vmask = torch.zeros(1, N, dtype=torch.bool)
    vmask[0, 2:32] = True
    lm._fastv_visual_mask = vmask
    n_vis = int(vmask.sum())
    M = (N - n_vis) + max(1, round(n_vis * 0.5))

    from transformers.cache_utils import DynamicCache
    cache = DynamicCache()
    ids = torch.randint(0, 300, (1, N))
    lm.model.forward(input_ids=ids, use_cache=True, past_key_values=cache, return_dict=True)

    # early layers cached full N, deep layers cached pruned M
    assert cache.get_seq_length(0) == N, cache.get_seq_length(0)
    assert cache.get_seq_length(K) == M, (cache.get_seq_length(K), M)
    assert cache.get_seq_length(nL - 1) == M

    # three decode steps; each grows every layer's cache by 1, no shape error
    for t in range(3):
        tok = torch.randint(0, 300, (1, 1))
        lm.model.forward(input_ids=tok, use_cache=True, past_key_values=cache,
                         return_dict=True)
        assert cache.get_seq_length(0) == N + t + 1
        assert cache.get_seq_length(K) == M + t + 1
    print(f"[ok] heterogeneous cache: early={cache.get_seq_length(0)} "
          f"deep={cache.get_seq_length(K)} after 3 decode steps (no shape errors)")


if __name__ == "__main__":
    test_build_keep_index()
    test_visual_mask_span()
    test_patched_prefill_prunes()
    test_decode_passthrough()
    test_prefill_then_decode_heterogeneous_cache()
    print("\nALL FASTV LOGIC TESTS PASSED")

"""
CPU integration test for gemma2_self_spec_decode against a tiny (but REAL)
Gemma2-style causal transformer -- exercises the actual
`transformers.cache_utils.HybridCache` (both its `_static_update` and
`_sliding_update` code paths) and the actual `DepthPruner`/`_BypassLayer`
classes, not a stateless mock.

Scope/limitation (stated explicitly, not hidden): `sliding_window` here is set
larger than any prompt+max_new_tokens used in these tests, so HybridCache's
sliding layers never hit the modulo-indexed wraparound/shift branch (only
relevant once total generated tokens exceed sliding_window). This matches the
real deployment regime (action sequences are ~12-30 tokens, far under a real
Gemma2 config's sliding_window, typically thousands) -- but the wraparound
edge case itself remains formally unverified. If a real SpatialVLA config used
a small sliding_window relative to max_new_tokens=256, this would need a
dedicated wraparound test before trusting the adapter.
"""
import types

import torch
import torch.nn as nn
from transformers.cache_utils import HybridCache

from depth_prune_gemma2 import DepthPruner
from gemma2_self_spec_decode import gemma2_self_speculative_generate

HIDDEN = 8
N_HEADS = 2
HEAD_DIM = HIDDEN // N_HEADS
N_LAYERS = 4
VOCAB = 16
EOS = VOCAB - 1
SLIDING_WINDOW = 64  # >> any test's prompt+max_new_tokens: no wraparound (see module docstring)


def make_config(seed):
    return types.SimpleNamespace(
        num_hidden_layers=N_LAYERS,
        num_attention_heads=N_HEADS,
        num_key_value_heads=N_HEADS,
        head_dim=HEAD_DIM,
        hidden_size=HIDDEN,
        sliding_window=SLIDING_WINDOW,
        sliding_window_pattern=2,  # layers 0,2 sliding; 1,3 global (matches Gemma2)
    )


class TinyGemma2Layer(nn.Module):
    def __init__(self, layer_idx, seed):
        super().__init__()
        self.layer_idx = layer_idx
        self.is_sliding = bool((layer_idx + 1) % 2)
        self.sliding_window = SLIDING_WINDOW if self.is_sliding else None
        g = torch.Generator().manual_seed(seed)
        self.q_proj = nn.Linear(HIDDEN, HIDDEN, bias=False)
        self.k_proj = nn.Linear(HIDDEN, HIDDEN, bias=False)
        self.v_proj = nn.Linear(HIDDEN, HIDDEN, bias=False)
        self.o_proj = nn.Linear(HIDDEN, HIDDEN, bias=False)
        self.mlp = nn.Sequential(nn.Linear(HIDDEN, HIDDEN * 2), nn.GELU(), nn.Linear(HIDDEN * 2, HIDDEN))
        for p in self.parameters():
            p.data = torch.randn(p.shape, generator=g) * 0.3

    def forward(self, hidden_states, attention_mask=None, position_ids=None,
                past_key_value=None, output_attentions=False, use_cache=False,
                cache_position=None, **kwargs):
        bsz, new_len, _ = hidden_states.shape
        residual = hidden_states
        q = self.q_proj(hidden_states).view(bsz, new_len, N_HEADS, HEAD_DIM).transpose(1, 2)
        k = self.k_proj(hidden_states).view(bsz, new_len, N_HEADS, HEAD_DIM).transpose(1, 2)
        v = self.v_proj(hidden_states).view(bsz, new_len, N_HEADS, HEAD_DIM).transpose(1, 2)

        if past_key_value is not None and use_cache:
            k, v = past_key_value.update(
                k, v, self.layer_idx,
                {"cache_position": cache_position, "sliding_window": self.sliding_window},
            )

        total_len = k.shape[2]
        # No wraparound in this test (see module docstring) -> cache slot index
        # == absolute position, so plain arange-based causal masking is valid.
        q_pos = cache_position.view(-1, 1)
        k_pos = torch.arange(total_len).view(1, -1)
        causal = (k_pos <= q_pos)
        scores = torch.matmul(q, k.transpose(-1, -2)) / (HEAD_DIM ** 0.5)
        scores = scores.masked_fill(~causal.view(1, 1, new_len, total_len), float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, v).transpose(1, 2).reshape(bsz, new_len, HIDDEN)
        hidden_states = residual + self.o_proj(out)
        hidden_states = hidden_states + self.mlp(hidden_states)

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (attn,)
        if use_cache:
            outputs += (past_key_value,)
        return outputs


class TinyGemma2Stack(nn.Module):
    def __init__(self, seed):
        super().__init__()
        self.layers = nn.ModuleList([TinyGemma2Layer(i, seed + i) for i in range(N_LAYERS)])

    def forward(self, inputs_embeds, attention_mask=None, past_key_values=None,
                use_cache=None, cache_position=None):
        h = inputs_embeds
        for layer in self.layers:
            out = layer(h, attention_mask=attention_mask, past_key_value=past_key_values,
                       use_cache=use_cache, cache_position=cache_position)
            h = out[0]
        return h


class TinyLanguageModel(nn.Module):
    """Mimics Gemma2ForCausalLM's surface DepthPruner needs: `.model.layers`, `.config`."""

    def __init__(self, seed):
        super().__init__()
        self.config = make_config(seed)
        self.model = TinyGemma2Stack(seed + 100)


class TinyVLAModel(nn.Module):
    """Mimics SpatialVLAForConditionalGeneration's forward surface: input_ids/
    pixel_values/intrinsic/attention_mask/past_key_values/use_cache in,
    .logits/.past_key_values out. pixel_values/intrinsic are accepted but
    unused (we're testing cache mechanics, not vision fusion)."""

    def __init__(self, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.embed = nn.Embedding(VOCAB, HIDDEN)
        self.embed.weight.data = torch.randn(VOCAB, HIDDEN, generator=g) * 0.3
        self.language_model = TinyLanguageModel(seed + 1000)
        self.lm_head = nn.Linear(HIDDEN, VOCAB, bias=False)
        self.lm_head.weight.data = torch.randn(VOCAB, HIDDEN, generator=g) * 0.3

    @property
    def dtype(self):
        return self.embed.weight.dtype

    def forward(self, input_ids=None, pixel_values=None, intrinsic=None,
                attention_mask=None, past_key_values=None, use_cache=None):
        h = self.embed(input_ids)
        if past_key_values is not None:
            past_seen = past_key_values.get_seq_length()
        else:
            past_seen = 0
        cache_position = torch.arange(past_seen, past_seen + h.shape[1])
        h = self.language_model.model(h, attention_mask=attention_mask,
                                      past_key_values=past_key_values,
                                      use_cache=use_cache, cache_position=cache_position)
        logits = self.lm_head(h)
        return _Out(logits=logits, past_key_values=past_key_values)


class _Out:
    def __init__(self, logits, past_key_values):
        self.logits = logits
        self.past_key_values = past_key_values


def make_pruner(vla):
    return DepthPruner(vla.language_model)


@torch.inference_mode()
def plain_greedy_generate(vla, input_ids, attention_mask, max_new_tokens, eos_token_id=None):
    """Reference: full model, one token at a time, no speculation, using a
    freshly-sized HybridCache exactly like gemma2_self_speculative_generate does."""
    prompt_len = input_ids.shape[1]
    cache = HybridCache(vla.language_model.config, max_batch_size=1,
                        max_cache_len=prompt_len + max_new_tokens,
                        device=input_ids.device, dtype=vla.dtype)
    out = vla(input_ids=input_ids, attention_mask=attention_mask, past_key_values=cache, use_cache=True)
    cache = out.past_key_values
    full_ids = input_ids
    am = attention_mask
    for _ in range(max_new_tokens):
        nxt = int(torch.argmax(out.logits[:, -1, :], dim=-1).item())
        full_ids = torch.cat([full_ids, torch.tensor([[nxt]])], dim=1)
        if eos_token_id is not None and nxt == eos_token_id:
            break
        am = torch.cat([am, torch.ones((1, 1), dtype=am.dtype)], dim=1)
        out = vla(input_ids=torch.tensor([[nxt]]), attention_mask=am, past_key_values=cache, use_cache=True)
    return full_ids


def _run_case(seed, gamma, max_new_tokens, draft_layers, prompt_len=3, cache_len_budget=64):
    vla = TinyVLAModel(seed=seed)
    pruner = make_pruner(vla)
    pruner.install_calibration_hooks()
    vla.language_model.model(vla.embed(torch.tensor([[1, 2, 3]])),
                             cache_position=torch.arange(3))  # populate redundancy stats (unused here, just for parity)
    pruner.finalize_calibration()

    import random
    rng = random.Random(seed + 999)
    prompt = torch.tensor([[rng.randrange(VOCAB) for _ in range(prompt_len)]])
    am = torch.ones_like(prompt)

    baseline = plain_greedy_generate(vla, prompt, am, max_new_tokens, eos_token_id=EOS)

    pruner2 = make_pruner(vla)  # fresh pruner sharing the SAME underlying layers/weights
    pruner2.ranking = list(range(N_LAYERS))  # not used directly; draft_layers passed explicitly
    spec = gemma2_self_speculative_generate(
        pruner2, vla, prompt, am, pixel_values=None, intrinsic=None,
        max_new_tokens=max_new_tokens, gamma=gamma, draft_layer_indices=draft_layers,
        eos_token_id=EOS, cache_len_budget=cache_len_budget,
    )
    return baseline, spec


def test_losslessness_various_seeds_and_gamma():
    for seed in range(12):
        for gamma in (1, 2, 3, 5):
            baseline, spec = _run_case(seed=seed, gamma=gamma, max_new_tokens=10, draft_layers=[1, 2])
            assert torch.equal(baseline, spec), (
                f"seed={seed} gamma={gamma}: baseline={baseline.tolist()} spec={spec.tolist()}"
            )
    print("ok: 12 seeds x 4 gamma values -> self-speculative output identical to plain greedy")


def test_losslessness_various_draft_layer_subsets():
    for draft_layers in ([1], [2], [3], [1, 2], [1, 2, 3], [1, 3]):
        baseline, spec = _run_case(seed=5, gamma=3, max_new_tokens=10, draft_layers=draft_layers)
        assert torch.equal(baseline, spec), f"draft_layers={draft_layers}: {baseline.tolist()} != {spec.tolist()}"
    print("ok: various draft_layers subsets (sliding + global layers) -> all identical")


def test_layer_0_never_bypassed_even_if_requested():
    # apply_indices must silently protect layer 0 regardless of what's asked
    vla = TinyVLAModel(seed=1)
    pruner = make_pruner(vla)
    pruned = pruner.apply_indices([0, 1, 2])
    assert 0 not in pruned, f"layer 0 leaked into bypass set: {pruned}"
    pruner.restore()
    print(f"ok: apply_indices protects layer 0 even when explicitly requested (pruned={pruned})")


def test_eos_matches():
    for seed in range(8):
        baseline, spec = _run_case(seed=seed, gamma=3, max_new_tokens=30, draft_layers=[1, 2])
        assert torch.equal(baseline, spec)
        if EOS in baseline[0].tolist():
            assert baseline.shape == spec.shape
    print("ok: EOS-terminated sequences (when they occur) match exactly")


def test_cache_budget_truncates_gracefully_when_too_small():
    # a tiny budget forces an early cutoff -- must not crash and must respect
    # the cap (a WARNING prints; that's checked visually, not asserted here).
    # effective budget is max(cache_len_budget, gamma+1) -- a floor so a round
    # always has room for at least one free token (see gemma2_self_spec_decode.py).
    gamma, cache_len_budget = 3, 3
    effective_budget = max(cache_len_budget, gamma + 1)
    baseline, spec = _run_case(seed=9, gamma=gamma, max_new_tokens=20, draft_layers=[1, 2],
                               cache_len_budget=cache_len_budget)
    prompt_len = 3
    new_len = spec.shape[1] - prompt_len
    assert new_len <= effective_budget, f"exceeded effective budget {effective_budget}: generated {new_len}"
    print(f"ok: cache_len_budget={cache_len_budget} (gamma={gamma}) truncates gracefully "
          f"at effective budget {effective_budget} ({new_len} new tokens, no crash)")


def test_generous_budget_matches_uncapped():
    # cache_len_budget >= max_new_tokens must be a no-op vs the uncapped path
    baseline, spec = _run_case(seed=4, gamma=3, max_new_tokens=10, draft_layers=[1, 2],
                               cache_len_budget=200)
    assert torch.equal(baseline, spec)
    print("ok: generous cache_len_budget (>> max_new_tokens) behaves identically to uncapped")


def test_stats_reported():
    vla = TinyVLAModel(seed=2)
    pruner = make_pruner(vla)
    prompt = torch.tensor([[1, 2, 3]])
    am = torch.ones_like(prompt)
    stats = {}
    spec = gemma2_self_speculative_generate(
        pruner, vla, prompt, am, pixel_values=None, intrinsic=None,
        max_new_tokens=15, gamma=4, draft_layer_indices=[1, 2], eos_token_id=EOS, stats=stats,
    )
    assert stats["rounds"] > 0
    assert stats["proposed"] >= stats["accepted"] >= 0
    print(f"ok: stats collected -> {stats}")


if __name__ == "__main__":
    test_losslessness_various_seeds_and_gamma()
    test_losslessness_various_draft_layer_subsets()
    test_layer_0_never_bypassed_even_if_requested()
    test_eos_matches()
    test_cache_budget_truncates_gracefully_when_too_small()
    test_generous_budget_matches_uncapped()
    test_stats_reported()
    print("\nALL PASS")

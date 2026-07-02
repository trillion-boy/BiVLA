"""
CPU integration test for `emu3_self_spec_decode.emu3_self_speculative_generate`
against a tiny (but REAL) causal transformer -- exercises the actual
`transformers.cache_utils.DynamicCache` API (`.update()`, `.crop()`) and the
actual `BypassDecoderLayer` class used in production, not a stateless mock.

This is the strongest correctness check possible without a GPU / real Emu3
checkpoint: if self-speculative decoding is lossless here, the *algorithm and
cache-surgery mechanics* are validated; what remains untested is only
Emu3-specific interface details (exact forward() kwargs, RoPE, etc.) that can
only be checked on the real model on Colab.
"""
import random

import torch
import torch.nn as nn
from transformers.cache_utils import DynamicCache

from bypass_layer import BypassDecoderLayer
from emu3_self_spec_decode import emu3_self_speculative_generate

VOCAB = 16
HIDDEN = 8
N_HEADS = 2
HEAD_DIM = HIDDEN // N_HEADS
N_LAYERS = 4
EOS = VOCAB - 1


class TinyDecoderLayer(nn.Module):
    def __init__(self, layer_idx, seed):
        super().__init__()
        self.layer_idx = layer_idx
        g = torch.Generator().manual_seed(seed)
        self.q_proj = nn.Linear(HIDDEN, HIDDEN, bias=False)
        self.k_proj = nn.Linear(HIDDEN, HIDDEN, bias=False)
        self.v_proj = nn.Linear(HIDDEN, HIDDEN, bias=False)
        self.o_proj = nn.Linear(HIDDEN, HIDDEN, bias=False)
        self.mlp = nn.Sequential(nn.Linear(HIDDEN, HIDDEN * 2), nn.GELU(), nn.Linear(HIDDEN * 2, HIDDEN))
        for p in self.parameters():
            p.data = torch.randn(p.shape, generator=g) * 0.3

    def forward(self, hidden_states, attention_mask=None, position_ids=None,
                past_key_value=None, output_attentions=False, use_cache=False, **kwargs):
        bsz, new_len, _ = hidden_states.shape
        residual = hidden_states
        q = self.q_proj(hidden_states).view(bsz, new_len, N_HEADS, HEAD_DIM).transpose(1, 2)
        k = self.k_proj(hidden_states).view(bsz, new_len, N_HEADS, HEAD_DIM).transpose(1, 2)
        v = self.v_proj(hidden_states).view(bsz, new_len, N_HEADS, HEAD_DIM).transpose(1, 2)

        past_len = 0
        if past_key_value is not None and use_cache:
            past_len = past_key_value.get_seq_length(self.layer_idx)
            k, v = past_key_value.update(k, v, self.layer_idx, {})

        total_len = k.shape[2]
        # causal mask: new-token i (absolute pos past_len+i) attends to <= past_len+i
        q_pos = torch.arange(past_len, past_len + new_len).view(-1, 1)
        k_pos = torch.arange(total_len).view(1, -1)
        causal = (k_pos <= q_pos)  # (new_len, total_len)
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


class TinyStack(nn.Module):
    def __init__(self, seed):
        super().__init__()
        self.layers = nn.ModuleList([TinyDecoderLayer(i, seed + i) for i in range(N_LAYERS)])

    def forward(self, inputs_embeds, attention_mask=None, past_key_values=None, use_cache=None):
        h = inputs_embeds
        for layer in self.layers:
            out = layer(h, attention_mask=attention_mask, past_key_value=past_key_values, use_cache=use_cache)
            h = out[0]
        return h


class TinyCausalLM(nn.Module):
    def __init__(self, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.embed = nn.Embedding(VOCAB, HIDDEN)
        self.embed.weight.data = torch.randn(VOCAB, HIDDEN, generator=g) * 0.3
        self.model = TinyStack(seed + 100)
        self.lm_head = nn.Linear(HIDDEN, VOCAB, bias=False)
        self.lm_head.weight.data = torch.randn(VOCAB, HIDDEN, generator=g) * 0.3

    def forward(self, input_ids=None, attention_mask=None, past_key_values=None, use_cache=None):
        h = self.embed(input_ids)
        h = self.model(h, attention_mask=attention_mask, past_key_values=past_key_values, use_cache=use_cache)
        logits = self.lm_head(h)
        return _Out(logits=logits, past_key_values=past_key_values)


class _Out:
    def __init__(self, logits, past_key_values):
        self.logits = logits
        self.past_key_values = past_key_values


class TinyPolicy:
    """Mimics the UniVLAInference surface emu3_self_speculative_generate needs."""

    def __init__(self, seed=0):
        self.model = TinyCausalLM(seed)
        self._original_decoder_layers = {}

    def _llm_decoder_layers(self):
        return self.model.model.layers

    def _restore_llm_layers(self):
        layers = self._llm_decoder_layers()
        for idx, layer in self._original_decoder_layers.items():
            layers[idx] = layer
        self._original_decoder_layers = {}

    def _apply_llm_pruning(self, layer_indices):
        layers = self._llm_decoder_layers()
        self._restore_llm_layers()
        for idx in layer_indices:
            self._original_decoder_layers[idx] = layers[idx]
            layers[idx] = BypassDecoderLayer(idx, num_kv_heads=N_HEADS, head_dim=HEAD_DIM)


class AllowedTokensProcessor:
    """Mirrors ActionIDConstraintLogitsProcessor: masks every token outside
    a fixed allowed set to -inf, regardless of input_ids (context-independent)."""

    def __init__(self, allowed_token_ids):
        self.allowed = set(allowed_token_ids)

    def __call__(self, input_ids, scores):
        mask = torch.zeros_like(scores, dtype=torch.bool)
        mask[:, list(self.allowed)] = True
        scores = scores.clone()
        scores[~mask] = -float("inf")
        return scores


@torch.inference_mode()
def plain_greedy_generate(policy, input_ids, attention_mask, max_new_tokens, eos_token_id=None,
                          logits_processor=None):
    """Reference: full model, one token at a time, no speculation."""
    def proc(ids, logits):
        if not logits_processor:
            return logits
        for p in logits_processor:
            logits = p(ids, logits)
        return logits

    policy._restore_llm_layers()
    cache = DynamicCache()
    out = policy.model(input_ids=input_ids, attention_mask=attention_mask, past_key_values=cache, use_cache=True)
    cache = out.past_key_values
    full_ids = input_ids
    am = attention_mask
    logits = proc(full_ids, out.logits[:, -1, :])
    for _ in range(max_new_tokens):
        nxt = int(torch.argmax(logits, dim=-1).item())
        full_ids = torch.cat([full_ids, torch.tensor([[nxt]])], dim=1)
        if eos_token_id is not None and nxt == eos_token_id:
            break
        am = torch.cat([am, torch.ones((1, 1), dtype=am.dtype)], dim=1)
        out = policy.model(input_ids=torch.tensor([[nxt]]), attention_mask=am, past_key_values=cache, use_cache=True)
        logits = proc(full_ids, out.logits[:, -1, :])
    return full_ids


def _run_case(seed, gamma, max_new_tokens, draft_layers, prompt_len=3):
    policy = TinyPolicy(seed=seed)
    rng = random.Random(seed + 999)
    prompt = torch.tensor([[rng.randrange(VOCAB) for _ in range(prompt_len)]])
    am = torch.ones_like(prompt)

    baseline = plain_greedy_generate(policy, prompt, am, max_new_tokens, eos_token_id=EOS)
    spec = emu3_self_speculative_generate(
        policy, prompt, am, max_new_tokens, gamma, draft_layers, eos_token_id=EOS,
    )
    return baseline, spec


def test_losslessness_various_seeds_and_gamma():
    for seed in range(15):
        for gamma in (1, 2, 3, 5):
            baseline, spec = _run_case(seed=seed, gamma=gamma, max_new_tokens=12, draft_layers=[1, 2])
            assert torch.equal(baseline, spec), (
                f"seed={seed} gamma={gamma}: baseline={baseline.tolist()} spec={spec.tolist()}"
            )
    print("ok: 15 seeds x 4 gamma values -> self-speculative output identical to plain greedy")


def test_losslessness_bypass_first_and_last_eligible_layers():
    # bypass layers near the edges of what's allowed (not layer 0, since that's
    # usually protected in practice, but exercise different subsets)
    for draft_layers in ([1], [2], [1, 2], [1, 2, 3]):
        baseline, spec = _run_case(seed=7, gamma=3, max_new_tokens=10, draft_layers=draft_layers)
        assert torch.equal(baseline, spec), f"draft_layers={draft_layers}: {baseline.tolist()} != {spec.tolist()}"
    print("ok: various draft_layers subsets -> all identical to plain greedy")


def test_losslessness_no_bypass_is_still_correct():
    # gamma>1 with an EMPTY draft set degenerates to "draft == full model" ->
    # every round should fully accept (draft is literally the same model).
    baseline, spec = _run_case(seed=3, gamma=4, max_new_tokens=10, draft_layers=[])
    assert torch.equal(baseline, spec)
    print("ok: draft_layers=[] (draft==full model) -> identical (sanity: full acceptance path)")


def test_losslessness_with_constrained_vocab_processor():
    # mirrors _generate_sequences's ActionIDConstraintLogitsProcessor: only a
    # subset of tokens are ever legal. Must be applied identically inside
    # spec-decode's draft/verify/correction/bonus, or output would diverge
    # from plain generation (and/or acceptance rate would collapse).
    allowed = [2, 5, 7, 9, EOS]
    for seed in range(10):
        policy = TinyPolicy(seed=seed)
        rng = random.Random(seed + 999)
        prompt = torch.tensor([[rng.randrange(VOCAB) for _ in range(3)]])
        am = torch.ones_like(prompt)
        proc = [AllowedTokensProcessor(allowed)]

        baseline = plain_greedy_generate(policy, prompt, am, 12, eos_token_id=EOS, logits_processor=proc)
        spec = emu3_self_speculative_generate(
            policy, prompt, am, 12, gamma=3, draft_layer_indices=[1, 2],
            eos_token_id=EOS, logits_processor=proc,
        )
        assert torch.equal(baseline, spec), f"seed={seed}: {baseline.tolist()} != {spec.tolist()}"
        # every generated token must be in the allowed set
        generated = spec[0, prompt.shape[1]:].tolist()
        assert all(t in allowed for t in generated), f"seed={seed}: {generated} has disallowed tokens"
    print("ok: constrained-vocab logits_processor respected, still identical to plain greedy")


def test_acceptance_stats_reported():
    policy = TinyPolicy(seed=1)
    prompt = torch.tensor([[1, 2, 3]])
    am = torch.ones_like(prompt)
    stats = {}
    spec = emu3_self_speculative_generate(policy, prompt, am, max_new_tokens=15, gamma=4,
                                          draft_layer_indices=[1, 2], eos_token_id=EOS, stats=stats)
    assert stats["rounds"] > 0
    assert stats["proposed"] >= stats["accepted"] >= 0
    print(f"ok: stats collected -> {stats}")


def test_eos_stops_and_matches():
    for seed in range(10):
        baseline, spec = _run_case(seed=seed, gamma=3, max_new_tokens=40, draft_layers=[1, 2])
        assert torch.equal(baseline, spec)
        if EOS in baseline[0].tolist():
            # baseline stops generation at EOS -- spec must match length exactly
            assert baseline.shape == spec.shape
    print("ok: EOS-terminated sequences (when they occur) match exactly")


if __name__ == "__main__":
    test_losslessness_various_seeds_and_gamma()
    test_losslessness_bypass_first_and_last_eligible_layers()
    test_losslessness_no_bypass_is_still_correct()
    test_losslessness_with_constrained_vocab_processor()
    test_acceptance_stats_reported()
    test_eos_stops_and_matches()
    print("\nALL PASS")

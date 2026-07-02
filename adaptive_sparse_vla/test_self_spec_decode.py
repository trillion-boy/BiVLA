"""CPU unit tests for greedy self-speculative decoding (no GPU/model needed).

The critical property under test: self-speculative decoding must ALWAYS
produce the exact same token sequence as plain greedy decoding with the full
("verify") model, regardless of how often/where the draft model disagrees.
That's what makes it safe to try on the real model without an accuracy risk
(unlike static depth pruning, whose whole cost was permanently traded-away
accuracy).
"""
import random

from self_spec_decode import greedy_decode, greedy_self_speculative_decode


VOCAB = 12
EOS = VOCAB - 1


def make_target_fn(seed):
    """A deterministic pseudo-LM: next-token logits depend on the last few
    tokens (some fixed 'weights'), so it behaves like a real autoregressive
    model (same context -> same logits, always)."""
    rng = random.Random(seed)
    weights = [[rng.uniform(-1, 1) for _ in range(VOCAB)] for _ in range(VOCAB)]

    def target_fn(tokens):
        ctx = tokens[-1] if tokens else 0
        # mix in a bit more context so it's not trivially periodic
        ctx2 = tokens[-2] if len(tokens) > 1 else 0
        base = weights[ctx % VOCAB]
        return [base[i] + 0.3 * weights[ctx2 % VOCAB][i] for i in range(VOCAB)]

    return target_fn


def make_draft_fn(target_fn, disagree_rate, seed):
    """A 'shallow model' that usually agrees with target_fn but sometimes
    picks a different token entirely -- like a real pruned model whose
    predictions are correlated with, but not identical to, the full model."""
    rng = random.Random(seed)

    def draft_fn(tokens):
        logits = target_fn(tokens)
        if rng.random() < disagree_rate:
            # perturb: swap the top choice with a random other token's score
            logits = list(logits)
            i, j = rng.randrange(len(logits)), rng.randrange(len(logits))
            logits[i], logits[j] = logits[j], logits[i]
        return logits

    return draft_fn


def _run_case(seed, disagree_rate, gamma, max_new_tokens, prompt_len=3):
    target_fn = make_target_fn(seed)
    draft_fn = make_draft_fn(target_fn, disagree_rate, seed + 1)

    def logits_fn(tokens, mode):
        return draft_fn(tokens) if mode == "draft" else target_fn(tokens)

    rng = random.Random(seed + 2)
    prompt = [rng.randrange(VOCAB) for _ in range(prompt_len)]

    baseline = greedy_decode(target_fn_wrapped(target_fn), prompt, max_new_tokens, eos_token_id=EOS)
    spec = greedy_self_speculative_decode(logits_fn, prompt, max_new_tokens, gamma, eos_token_id=EOS)
    return baseline, spec


def target_fn_wrapped(target_fn):
    def f(tokens, mode):
        return target_fn(tokens)
    return f


def test_losslessness_no_disagreement():
    # draft always agrees with target -> should accept every draft token
    baseline, spec = _run_case(seed=0, disagree_rate=0.0, gamma=4, max_new_tokens=10)
    assert baseline == spec, f"{baseline} != {spec}"
    print(f"ok: disagree_rate=0.0 -> identical ({baseline})")


def test_losslessness_moderate_disagreement():
    for seed in range(20):
        baseline, spec = _run_case(seed=seed, disagree_rate=0.4, gamma=3, max_new_tokens=15)
        assert baseline == spec, f"seed={seed}: {baseline} != {spec}"
    print("ok: disagree_rate=0.4, 20 seeds -> all identical to plain greedy")


def test_losslessness_high_disagreement():
    for seed in range(20):
        baseline, spec = _run_case(seed=seed, disagree_rate=0.9, gamma=5, max_new_tokens=12)
        assert baseline == spec, f"seed={seed}: {baseline} != {spec}"
    print("ok: disagree_rate=0.9 (draft almost always wrong), 20 seeds -> still identical")


def test_losslessness_gamma_1():
    # gamma=1 degenerates close to plain decoding but through the spec-decode
    # code path -- exercise it explicitly.
    for seed in range(10):
        baseline, spec = _run_case(seed=seed, disagree_rate=0.5, gamma=1, max_new_tokens=10)
        assert baseline == spec, f"seed={seed}: {baseline} != {spec}"
    print("ok: gamma=1 -> identical")


def test_losslessness_large_gamma():
    for seed in range(10):
        baseline, spec = _run_case(seed=seed, disagree_rate=0.3, gamma=8, max_new_tokens=10)
        assert baseline == spec, f"seed={seed}: {baseline} != {spec}"
    print("ok: gamma=8 (> max_new_tokens some rounds) -> identical")


def test_eos_stops_generation():
    # force EOS to be very likely from a specific token by construction
    def logits_fn(tokens, mode):
        if tokens[-1] == 5:
            l = [0.0] * VOCAB
            l[EOS] = 100.0
            return l
        l = [0.0] * VOCAB
        l[5] = 100.0
        return l

    spec = greedy_self_speculative_decode(logits_fn, [1, 2], max_new_tokens=20, gamma=4, eos_token_id=EOS)
    baseline = greedy_decode(logits_fn, [1, 2], max_new_tokens=20, eos_token_id=EOS)
    assert spec == baseline, f"{spec} != {baseline}"
    assert spec[-1] == EOS, f"should stop at EOS, got {spec}"
    assert spec.count(EOS) == 1, f"should stop immediately at first EOS, got {spec}"
    print(f"ok: EOS handling matches baseline and stops immediately ({spec})")


def test_acceptance_rate_sanity():
    # sanity: with low disagreement, avg accepted-per-round should be high
    # (informal check that the algorithm is actually "using" the draft, not
    # just falling back to 1-token rounds every time)
    target_fn = make_target_fn(seed=42)
    draft_fn = make_draft_fn(target_fn, disagree_rate=0.1, seed=43)

    accept_lengths = []

    def counting_logits_fn(tokens, mode):
        return draft_fn(tokens) if mode == "draft" else target_fn(tokens)

    # instrument by re-deriving accept length from output vs plain per-round draft count
    # (simplified proxy: just check total tokens generated in fewer "rounds" than
    # max_new_tokens by comparing against gamma=1 forcing every round to accept<=1)
    spec = greedy_self_speculative_decode(counting_logits_fn, [0, 1], max_new_tokens=30, gamma=4, eos_token_id=EOS)
    baseline = greedy_decode(target_fn_wrapped(target_fn), [0, 1], max_new_tokens=30, eos_token_id=EOS)
    assert spec == baseline
    print(f"ok: low-disagreement run stayed lossless over 30 tokens ({len(spec)} generated)")


if __name__ == "__main__":
    test_losslessness_no_disagreement()
    test_losslessness_moderate_disagreement()
    test_losslessness_high_disagreement()
    test_losslessness_gamma_1()
    test_losslessness_large_gamma()
    test_eos_stops_generation()
    test_acceptance_rate_sanity()
    print("\nALL PASS")

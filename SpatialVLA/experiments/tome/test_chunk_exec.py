"""CPU unit tests for chunk_exec queue mechanics (no model needed)."""
import torch

from chunk_exec import apply_chunk_execution, reset_chunk_execution, remove_chunk_execution

CHUNK = 4  # 4 actions x 3 tokens = 12 tokens per generate


class FakeVLA:
    def __init__(self):
        self.generate_count = 0

    def predict_action(self, model_inputs):
        # 12 action tokens: action i = [100*g+i*10+1, ..+2, ..+3] so every token
        # is traceable to (generate call g, action index i)
        g = self.generate_count
        self.generate_count += 1
        toks = []
        for i in range(CHUNK):
            toks += [1000 * g + 10 * i + 1, 1000 * g + 10 * i + 2, 1000 * g + 10 * i + 3]
        return torch.tensor([toks])


class FakeProcessor:
    action_chunk_size = CHUNK


class FakePolicy:
    def __init__(self):
        self.vla = FakeVLA()
        self.processor = FakeProcessor()


def first_action(tokens_1xN):
    return tokens_1xN[0, :3].tolist()


def test_full_chunk_execution():
    policy = FakePolicy()
    st = apply_chunk_execution(policy, k=4, verbose=False)
    outs = [policy.vla.predict_action(None) for _ in range(8)]
    # 8 steps -> only 2 real generates
    assert policy.vla.generate_count == 2, policy.vla.generate_count
    assert st["gen_calls"] == 2 and st["steps"] == 8
    # wrapper reads actions[0] of each output -> must be actions 0,1,2,3 of gen0 then gen1
    got = [first_action(o) for o in outs]
    want = [[10 * i + 1, 10 * i + 2, 10 * i + 3] for i in range(4)] + \
           [[1000 + 10 * i + 1, 1000 + 10 * i + 2, 1000 + 10 * i + 3] for i in range(4)]
    assert got == want, f"{got} != {want}"
    # popped outputs are tiled to full chunk width so decode_actions sees 3*CHUNK tokens
    assert all(o.shape == (1, 3 * CHUNK) for o in outs[1:4])
    print("ok: k=4 -> 2 generates for 8 steps, actions executed in exact predicted order")


def test_partial_k():
    policy = FakePolicy()
    apply_chunk_execution(policy, k=2, verbose=False)
    outs = [policy.vla.predict_action(None) for _ in range(6)]
    # k=2 -> generate every 2 steps -> 3 generates
    assert policy.vla.generate_count == 3
    got = [first_action(o) for o in outs]
    want = []
    for g in range(3):
        for i in range(2):
            want.append([1000 * g + 10 * i + 1, 1000 * g + 10 * i + 2, 1000 * g + 10 * i + 3])
    assert got == want, f"{got} != {want}"
    print("ok: k=2 -> only first 2 of 4 predicted actions executed, then replan")


def test_reset_flushes_queue():
    policy = FakePolicy()
    apply_chunk_execution(policy, k=4, verbose=False)
    policy.vla.predict_action(None)          # gen0, queue = actions 1..3
    reset_chunk_execution(policy)            # new episode -> flush
    out = policy.vla.predict_action(None)    # must be a REAL generate (gen1)
    assert policy.vla.generate_count == 2
    assert first_action(out) == [1001, 1002, 1003]
    print("ok: reset flushes queue -> next step re-generates (episode boundary safe)")


def test_remove_restores():
    policy = FakePolicy()
    apply_chunk_execution(policy, k=4, verbose=False)
    remove_chunk_execution(policy)
    [policy.vla.predict_action(None) for _ in range(3)]
    assert policy.vla.generate_count == 3, "after remove, every step must re-generate"
    print("ok: remove restores original per-step generation")


def test_no_chunk_model_disables():
    policy = FakePolicy()
    policy.processor.action_chunk_size = 1
    st = apply_chunk_execution(policy, k=4, verbose=False)
    assert st is None
    print("ok: chunk_size=1 model -> cleanly disabled")


if __name__ == "__main__":
    test_full_chunk_execution()
    test_partial_k()
    test_reset_flushes_queue()
    test_remove_restores()
    test_no_chunk_model_disables()
    print("\nALL PASS")

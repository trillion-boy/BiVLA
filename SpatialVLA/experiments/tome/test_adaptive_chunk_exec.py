"""Tests for adaptive chunk execution."""
import torch
from chunk_exec import apply_chunk_execution, reset_chunk_execution, remove_chunk_execution
from adaptive_chunk_exec import AdaptiveChunkExecutor


class FakeVLA:
    def __init__(self):
        self.generate_count = 0

    def predict_action(self, model_inputs):
        g = self.generate_count
        self.generate_count += 1
        toks = []
        for i in range(4):  # 4-action chunk
            toks += [1000 * g + 10 * i + 1, 1000 * g + 10 * i + 2, 1000 * g + 10 * i + 3]
        return torch.tensor([toks])


class FakeProcessor:
    action_chunk_size = 4


class FakePolicy:
    def __init__(self):
        self.vla = FakeVLA()
        self.processor = FakeProcessor()


def test_adaptive_starts_sparse():
    """Adaptive executor should start with k_sparse (k=4)."""
    policy = FakePolicy()
    executor = AdaptiveChunkExecutor(policy, k_sparse=4, k_dense=1)
    assert executor.current_k == 4
    state = executor.get_state()
    assert state["current_k"] == 4
    print("ok: adaptive starts with k_sparse=4")


def test_gripper_close_triggers_dense():
    """When gripper closes, should switch to k_dense after delay."""
    policy = FakePolicy()
    executor = AdaptiveChunkExecutor(policy, k_sparse=4, k_dense=1,
                                      close_steps_before_dense=2)
    # Initial state: k=4, gripper open
    assert executor.current_k == 4

    # Step 1: gripper closes
    action1 = {"gripper": 0.2}  # closed
    executor.step(action1)
    assert executor.current_k == 4  # Still sparse (just closed, steps_since=0)

    # Step 2: gripper still closed
    action2 = {"gripper": 0.1}
    executor.step(action2)
    assert executor.current_k == 4  # Still sparse (steps_since=1 < 2)

    # Step 3: gripper still closed
    action3 = {"gripper": 0.1}
    executor.step(action3)
    assert executor.current_k == 1  # Now dense (steps_since=2 >= 2)
    print("ok: gripper close triggers switch to dense after delay")


def test_gripper_open_triggers_sparse():
    """When gripper opens, should switch back to sparse after delay."""
    policy = FakePolicy()
    executor = AdaptiveChunkExecutor(policy, k_sparse=4, k_dense=1,
                                      close_steps_before_dense=1,
                                      open_steps_after_release=2)
    # Start with closed gripper in dense mode
    action0 = {"gripper": 0.2}
    executor.step(action0)  # steps_since=0
    executor.step(action0)  # steps_since=1 >= 1 → switch to dense
    assert executor.current_k == 1  # Dense

    # Step 1: gripper opens
    action1 = {"gripper": 0.8}
    executor.step(action1)  # gripper opens, steps_since=0
    assert executor.current_k == 1  # Still dense (just opened)

    # Step 2: gripper still open
    action2 = {"gripper": 0.9}
    executor.step(action2)  # steps_since=1 < 2
    assert executor.current_k == 1  # Still dense (steps_since=1)

    # Step 3: gripper still open
    action3 = {"gripper": 0.9}
    executor.step(action3)  # steps_since=2 >= 2 → switch to sparse
    assert executor.current_k == 4  # Back to sparse (waited enough)
    print("ok: gripper open triggers switch to sparse after delay")


def test_reset_flushes_queue():
    """Reset should flush chunk queue and reset to sparse."""
    policy = FakePolicy()
    executor = AdaptiveChunkExecutor(policy, k_sparse=4, k_dense=1,
                                      close_steps_before_dense=1)
    # Put into dense mode
    action = {"gripper": 0.2}
    executor.step(action)
    executor.step(action)
    assert executor.current_k == 1

    # Reset episode
    executor.reset()
    assert executor.current_k == 4  # Back to sparse
    assert executor.episode_step == 0
    assert executor.gripper_closed == False
    print("ok: reset flushes queue and resets to sparse")


def test_multiple_cycles():
    """Test multiple grasp/release cycles."""
    policy = FakePolicy()
    executor = AdaptiveChunkExecutor(policy, k_sparse=4, k_dense=1,
                                      close_steps_before_dense=1,
                                      open_steps_after_release=1)
    k_values = []

    # Initial state
    k_values.append(executor.current_k)  # Should be 4

    # Cycle 1: close gripper (grasp)
    action_close = {"gripper": 0.2}
    executor.step(action_close)  # steps_since=0
    k_values.append(executor.current_k)  # Still 4
    executor.step(action_close)  # steps_since=1 >= 1 → switch to dense
    k_values.append(executor.current_k)  # Should be 1 (dense)

    # Cycle 2: open gripper (release)
    action_open = {"gripper": 0.9}
    executor.step(action_open)  # steps_since=0
    k_values.append(executor.current_k)  # Still 1
    executor.step(action_open)  # steps_since=1 >= 1 → switch to sparse
    k_values.append(executor.current_k)  # Should be 4 (sparse)

    # Cycle 3: close gripper again
    executor.step(action_close)  # steps_since=0
    k_values.append(executor.current_k)  # Still 4
    executor.step(action_close)  # steps_since=1 >= 1 → switch to dense
    k_values.append(executor.current_k)  # Should be 1 (dense)

    # Expected pattern: [4, 4, 1, 1, 4, 4, 1]
    expected = [4, 4, 1, 1, 4, 4, 1]
    assert k_values == expected, f"got {k_values}, expected {expected}"
    print("ok: multiple grasp/release cycles work correctly")


if __name__ == "__main__":
    test_adaptive_starts_sparse()
    test_gripper_close_triggers_dense()
    test_gripper_open_triggers_sparse()
    test_reset_flushes_queue()
    test_multiple_cycles()
    print("\nALL ADAPTIVE CHUNK TESTS PASS")

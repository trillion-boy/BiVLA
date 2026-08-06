"""Pin the Google Robot gripper convention, and pin that Bridge did not move.

The gripper convention is the one part of Fractal support that fails silently.
Get it wrong and nothing raises: the arm still moves, episodes still terminate,
and the harness prints a low-but-plausible success rate that reads exactly like
a real result. So it is checked here rather than trusted.

Three things are asserted:

1. WidowX is UNCHANGED. Every OpenVLA/Bridge number already in
   `experiments/OpenVLA_Bridge_Grid.md` was produced by the pre-Fractal code
   path, and adding a second embodiment must not silently re-measure them.
   The Bridge gripper is absolute and stateless, so this is exact.
2. Google Robot latches. Its gripper takes a RELATIVE command and one step of
   "close" does not physically close the fingers, so SimplerEnv's convention
   re-issues the transition for `sticky_gripper_num_repeat` env steps.
3. The latch is measured in ENV steps, not model calls. Under action repeat k
   the two differ by k, and only the per-env-step reading keeps the latch the
   same physical duration at every k. Otherwise the k=4 condition would hold
   the gripper transition 4x as long as the baseline and the horizon curve
   would be measuring that instead of the horizon.

    python test_google_robot_gripper.py
    # -> ALL GOOGLE ROBOT GRIPPER CHECKS PASSED
"""
import os
import sys
import types

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

# transforms3d is the only import `openvla_inference` needs that a plain CI box
# may lack, and this test never exercises the rotation path. Stubbing it keeps
# the gripper convention testable without a full robotics install; if the real
# package is present it is used instead.
try:
    import transforms3d  # noqa: F401
except ImportError:
    _t3d = types.ModuleType("transforms3d")
    _euler = types.ModuleType("transforms3d.euler")
    _euler.euler2axangle = lambda r, p, y: (np.array([0.0, 0.0, 1.0]), 0.0)
    _t3d.euler = _euler
    sys.modules["transforms3d"] = _t3d
    sys.modules["transforms3d.euler"] = _euler

from openvla_inference import (  # noqa: E402
    POLICY_SETUPS,
    ActionRepeatOpenVLAInference,
    OpenVLAInference,
)


def make(cls, policy_setup, **extra):
    """An inference object with its conventions set and no model loaded.

    `__init__` downloads a 7B checkpoint, which this test has no use for. The
    gripper convention lives entirely in the small amount of state below.
    """
    obj = object.__new__(cls)
    obj.policy_setup = policy_setup
    obj.sticky_gripper_num_repeat = POLICY_SETUPS[policy_setup]["sticky_gripper_num_repeat"]
    obj._reset_gripper_state()
    for k, v in extra.items():
        setattr(obj, k, v)
    return obj


def grip(model, opens):
    """-> the gripper command for each of a sequence of raw `open_gripper` values."""
    return [float(model._gripper_command(o)[0]) for o in opens]


def check_bridge_is_unchanged():
    """WidowX: absolute, stateless, and identical to the pre-Fractal code."""
    m = make(OpenVLAInference, "widowx_bridge")
    raw = [0.9, 0.9, 0.1, 0.1, 0.9, 0.2, 0.8]
    got = grip(m, raw)
    # This is verbatim the expression the pre-Fractal `transform_action` used.
    want = [1.0 if r > 0.5 else -1.0 for r in raw]
    assert got == want, f"Bridge gripper changed: {got} != {want}"

    # Stateless means order does not matter and repetition is a no-op -- which
    # is what makes action repeat on Bridge unaffected by this change.
    assert grip(make(OpenVLAInference, "widowx_bridge"), [0.1] * 5) == [-1.0] * 5
    assert POLICY_SETUPS["widowx_bridge"]["sticky_gripper_num_repeat"] == 1
    assert POLICY_SETUPS["widowx_bridge"]["unnorm_key"] == "bridge_orig"
    print("  bridge: absolute, stateless, unchanged")


def check_google_robot_latches():
    """Google Robot: relative command, transition held for N env steps."""
    n = POLICY_SETUPS["google_robot"]["sticky_gripper_num_repeat"]
    assert n == 15, f"expected SimplerEnv's OpenVLA setting of 15, got {n}"
    assert POLICY_SETUPS["google_robot"]["unnorm_key"] == "fractal20220817_data"

    m = make(OpenVLAInference, "google_robot")
    # First call has no previous action to difference against -> no motion.
    assert grip(m, [0.9]) == [0.0], "first step must not command a transition"

    # Open (+1) -> close (-1) is a relative delta of +2, held for n steps.
    got = grip(m, [0.1] * (n + 3))
    assert got[:n] == [2.0] * n, f"latch should hold {n} steps, got {got[:n]}"
    assert got[n:] == [0.0] * 3, f"latch should release after {n} steps, got {got[n:]}"

    # And it re-arms on the reverse transition.
    got = grip(m, [0.9] * n)
    assert got == [-2.0] * n, f"reverse transition should latch too, got {got}"
    print(f"  google_robot: relative, latched for {n} env steps, re-arms")


def check_latch_is_per_env_step_not_per_model_call():
    """The latch must last the same wall-clock at every action-repeat k.

    This is the assertion that makes the Fractal horizon curve mean anything:
    the k=1, k=2 and k=4 conditions must differ in how often the POLICY is
    consulted, not in how long the gripper transition is held.
    """
    n = POLICY_SETUPS["google_robot"]["sticky_gripper_num_repeat"]
    raw_open = np.array([0.0] * 6 + [0.9], dtype=np.float32)
    raw_close = np.array([0.0] * 6 + [0.1], dtype=np.float32)

    per_k = {}
    for k in (1, 2, 4):
        m = make(ActionRepeatOpenVLAInference, "google_robot", repeat_k=k,
                 _last_prepared_image=None)
        env_steps = []
        # One "open" model call, then "close" calls until well past the latch.
        for raw in [raw_open] + [raw_close] * (2 * n // k + 4):
            env_steps += [float(a["gripper"][0]) for a in
                          (m._env_action_from_raw(raw) for _ in range(k))]
        # How many ENV steps carried the close transition.
        per_k[k] = sum(1 for g in env_steps if g == 2.0)

    assert set(per_k.values()) == {n}, (
        f"latch duration must be {n} env steps at every action repeat, got {per_k}"
    )
    print(f"  latch spans {n} env steps at k=1, 2 and 4 alike")


def check_reset_clears_the_latch():
    """A half-finished close must not leak into the next episode."""
    m = make(OpenVLAInference, "google_robot")
    grip(m, [0.9, 0.1, 0.1])           # mid-latch
    assert m._sticky_on, "precondition: the latch should be armed here"
    m._last_prepared_image = None
    m.reset()
    assert not m._sticky_on and m._prev_gripper is None, "reset must clear the latch"
    assert grip(m, [0.1]) == [0.0], "a fresh episode starts with no transition"
    print("  reset clears the latch")


def main():
    check_bridge_is_unchanged()
    check_google_robot_latches()
    check_latch_is_per_env_step_not_per_model_call()
    check_reset_clears_the_latch()
    print("ALL GOOGLE ROBOT GRIPPER CHECKS PASSED")


if __name__ == "__main__":
    main()

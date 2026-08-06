"""Pin the Google Robot protocol: distinct states, and one table for everyone.

The protocol is a mapping from an episode index to one fixed initial state. Two
things about it are load-bearing and neither shows up in a result if it breaks:

* Every index in a task's range must map to a DIFFERENT state. If two indices
  collide, n goes up while information does not, and the success rate is a
  weighted average of fewer states than it claims.
* Every harness must resolve the SAME table. The cross-backbone comparison is
  "OpenVLA and SpatialVLA on identical initial states"; two harnesses that each
  hold their own copy would drift by one `max_episode_steps` and still produce
  numbers that look comparable.

    python test_simpler_fractal_protocol.py
    # -> ALL FRACTAL PROTOCOL CHECKS PASSED
"""
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import simpler_fractal_protocol as proto


def key(opts):
    """A hashable identity for a reset option dict."""
    return json.dumps(opts, sort_keys=True, default=float)


def check_every_episode_is_a_distinct_state():
    for task, cfg in proto.GOOGLE_ROBOT_TASKS.items():
        lo, hi = cfg["obj_episode_range"]
        seeds, states = [], []
        for ep in range(lo, hi):
            seed, opts = proto.prepackaged_reset_options(cfg, ep)
            seeds.append(seed)
            states.append(key(opts))
        assert len(set(seeds)) == hi - lo, f"{task}: seeds collide"
        if cfg["variation"] == "seed_only":
            # The seed IS the variation here -- the env draws its station,
            # drawer and URDF from it -- so identical option dicts are correct.
            assert set(states) == {"{}"}, f"{task}: seed_only should pass no options"
        else:
            assert len(set(states)) == hi - lo, (
                f"{task}: {hi - lo} episodes map to only {len(set(states))} states"
            )
    print(f"  {len(proto.GOOGLE_ROBOT_TASKS)} tasks: every episode is a distinct state")


def check_the_grid_refuses_to_overrun():
    """Asking past the protocol must raise, not silently wrap and inflate n."""
    cfg = proto.GOOGLE_ROBOT_TASKS["google_robot_pick_standing_coke_can"]
    n = cfg["obj_episode_range"][1]
    proto.prepackaged_reset_options(cfg, n - 1)          # the last real state
    try:
        proto.prepackaged_reset_options(cfg, n)
    except ValueError:
        print(f"  xy_grid refuses episode {n} (protocol defines {n})")
        return
    raise AssertionError("running past the grid should raise, not wrap")


def check_move_near_variants_are_registered_separately():
    """v0 and v1 are different envs and must never be pooled in one table."""
    v0 = proto.GOOGLE_ROBOT_TASKS["google_robot_move_near_v0"]["env_name"]
    v1 = proto.GOOGLE_ROBOT_TASKS["google_robot_move_near"]["env_name"]
    assert v0.endswith("-v0") and v1.endswith("-v1") and v0 != v1
    print(f"  move_near variants distinct: {v0} vs {v1}")


def check_both_harnesses_resolve_one_table():
    """The path-walk in each harness must find THIS module, not a copy."""
    found = 0
    for rel in ("SpatialVLA/experiments/latent_saccade/spatialvla_eval.py",
                "RetinaBased/PythonProject/simple_eval.py"):
        path = os.path.join(_HERE, rel)
        if not os.path.exists(path):
            continue
        src = open(path).read()
        assert "_load_fractal_protocol" in src, f"{rel} does not import the shared protocol"
        assert "GOOGLE_ROBOT_TASKS" in src, f"{rel} does not use the shared table"
        # The giveaway that someone re-inlined a copy.
        assert "GraspSingleOpenedCokeCanInScene" not in src, (
            f"{rel} holds its own copy of the Google Robot task table"
        )
        found += 1
    assert found, "neither harness was found to check"
    print(f"  {found} harness(es) import the shared table, none hold a copy")


def main():
    check_every_episode_is_a_distinct_state()
    check_the_grid_refuses_to_overrun()
    check_move_near_variants_are_registered_separately()
    check_both_harnesses_resolve_one_table()
    print("ALL FRACTAL PROTOCOL CHECKS PASSED")


if __name__ == "__main__":
    main()

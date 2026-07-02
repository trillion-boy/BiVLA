"""
Action-chunk execution for SpatialVLA: call the expensive generate 1/k as often.

Discovery (processing_spatialvla.py::decode_actions): one action = exactly 3
tokens (translation + rotation + gripper), and the processor decodes
`3 * action_chunk_size` tokens into a CHUNK of future actions per generate
call. The profiler measured ~12 decode tokens/step = a chunk of 4 actions --
the model already PAYS for 4 future actions every control step, but the
SimplerEnv wrapper uses only actions[0] and re-generates from scratch next
step, discarding 3/4 of what was computed.

This module executes the chunk instead: patch `vla.predict_action` so that
after each real generate, the remaining k-1 actions are queued; the next k-1
wrapper calls pop from the queue with NO model call at all (skipping ViT
encode + prefill + decode entirely). Decode latency per env step is amortized
by ~k. Training-free, no external module, and -- unlike every "make decode
cheaper" attempt on this backbone (layer pruning, spec-decode) -- it does not
touch the model's computation at all, so there is no accuracy mechanism to
break: the actions executed are exactly the ones the model predicted as a
coherent k-step plan (chunked prediction is what the checkpoint was
configured for; the wrapper's discard-and-replan was a choice, not a
requirement).

Tradeoff to measure honestly: executing actions 2..k open-loop means less
frequent visual feedback (like temporal caching, but at the action level).
Stride-2 feature caching held success while stride-3 collapsed -- so expect a
sweet spot in k; measure success alongside latency. The next iteration is the
ADAPTIVE version (replan densely near grasp, sparsely in transport -- the
project's non-uniform-over-time thesis applied to replanning frequency).

Queue mechanics (popped actions round-trip through the WRAPPER's own
decode/convert path, so gripper sticky logic etc. still runs per step):
a popped 3-token action is returned tiled to the full (1, 3*chunk) shape
(identical copies); decode_actions decodes them all and the wrapper takes
actions[0] = the popped one. No padding warnings, no wrapper changes.
"""
from __future__ import annotations


def apply_chunk_execution(policy, k: int = 0, verbose: bool = True):
    """Patch policy's underlying vla so only every k-th wrapper step runs the
    real generate. k=0 or k>chunk_size -> use the model's full chunk size.
    Returns the state dict (for stats) or None if the model has no chunk."""
    vla = getattr(policy, "vla", None) or getattr(policy, "model", None)
    proc = getattr(policy, "processor", None)
    chunk_size = int(getattr(proc, "action_chunk_size", 1) or 1)
    if chunk_size <= 1:
        if verbose:
            print("[ChunkExec] processor.action_chunk_size<=1: model predicts no "
                  "chunk, nothing to execute -- disabled.", flush=True)
        return None
    if k <= 0 or k > chunk_size:
        k = chunk_size

    if getattr(vla, "_chunk_exec_state", None):
        remove_chunk_execution(policy)

    state = {"queue": [], "k": k, "chunk_size": chunk_size,
             "orig": vla.predict_action, "gen_calls": 0, "steps": 0}

    def patched_predict_action(*args, **kwargs):
        state["steps"] += 1
        if state["queue"]:
            triple = state["queue"].pop(0)          # (1, 3) tokens of one action
            return triple.repeat(1, chunk_size)      # tiled; wrapper uses actions[0]
        out = state["orig"](*args, **kwargs)         # (1, n_new_tokens) real generate
        state["gen_calls"] += 1
        toks = out[:, : 3 * chunk_size]
        n_actions = toks.shape[1] // 3
        triples = [toks[:, i * 3:(i + 1) * 3] for i in range(min(k, n_actions))]
        state["queue"] = triples[1:]                 # actions 2..k for the next steps
        return out

    vla.predict_action = patched_predict_action
    vla._chunk_exec_state = state
    if verbose:
        print(f"[ChunkExec] executing {k}/{chunk_size} predicted actions per generate "
              f"(model called every {k} env steps; decode amortized ~{k}x).", flush=True)
    return state


def reset_chunk_execution(policy):
    """Flush the queue (call per episode, and later: on adaptive replan triggers)."""
    vla = getattr(policy, "vla", None) or getattr(policy, "model", None)
    st = getattr(vla, "_chunk_exec_state", None)
    if st:
        st["queue"] = []
    return policy


def remove_chunk_execution(policy):
    vla = getattr(policy, "vla", None) or getattr(policy, "model", None)
    st = getattr(vla, "_chunk_exec_state", None)
    if st:
        vla.predict_action = st["orig"]
        vla._chunk_exec_state = None
    return policy

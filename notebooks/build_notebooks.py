"""Build the four per-method handoff notebooks.

Written as a generator rather than four hand-edited .ipynb files because the
loop diagram and the hook table appear in all four and must not drift: a
handoff notebook whose "where to apply it" section disagrees with the other
three is worse than no notebook.
"""
import json
import os
import textwrap

OUT = "/home/user/BiVLA/notebooks"
os.makedirs(OUT, exist_ok=True)


def md(text):
    return {"cell_type": "markdown", "metadata": {},
            "source": textwrap.dedent(text).strip("\n").splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": textwrap.dedent(text).strip("\n").splitlines(keepends=True)}


def _check_markdown(name, cells):
    """Refuse to write a markdown cell that renders as a preformatted block.

    Markdown treats a 4-space-indented line as code. These cells are authored
    as indented triple-quoted strings and dedented on the way out, so a single
    line left at column 0 makes textwrap.dedent find no common prefix, strip
    nothing, and turn the whole cell into a grey box. It is invisible in the
    source and only shows up when the notebook is opened, which is exactly why
    it is checked here instead of by eye.
    """
    for i, c in enumerate(cells):
        if c["cell_type"] != "markdown":
            continue
        in_fence = False
        for ln, line in enumerate("".join(c["source"]).split("\n")):
            if line.strip().startswith("```"):
                in_fence = not in_fence
                continue
            if in_fence or not line.strip():
                continue
            if line.startswith("    "):
                raise SystemExit(
                    f"{name} cell {i} line {ln}: indented prose outside a code "
                    f"fence -- this cell would render as a code block.\n"
                    f"  {line[:70]}\n"
                    f"  (usually one line left at column 0 defeats the dedent)")


def write(name, cells):
    _check_markdown(name, cells)
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python",
                           "name": "python3"},
            "language_info": {"name": "python", "version": "3.10"},
            "colab": {"provenance": []},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path = os.path.join(OUT, name)
    with open(path, "w") as fh:
        json.dump(nb, fh, indent=1)
        fh.write("\n")
    print("wrote", path)


# ---------------------------------------------------------------- shared text

LOOP_DIAGRAM = """
```
                    ┌─────────────────────────────────────────┐
                    │                                         │
              ┌─────▼──────┐                                  │
              │  observe   │   obs = env.step(action)         │
              └─────┬──────┘                                  │
                    │  raw camera frame (H, W, 3) uint8       │
        ╔═══════════▼═══════════╗                             │
        ║  HOOK A                ║  ← 02 foveation            │
        ║  transform the image   ║                             │
        ╚═══════════╤═══════════╝                             │
                    │                                         │
              ┌─────▼──────────────────────────────┐          │
              │  policy.step(image, instruction)   │          │
              │                                    │          │
              │   ├─ preprocess (resize/normalise) │          │
              │   ├─ vision encoder                │          │
              │   ├─ LLM decoder stack ────────────┼──╗       │
              │   └─ action de-tokenise            │  ║       │
              └─────┬──────────────────────────────┘  ║       │
                    │  actions (T, action_dim)        ║       │
        ╔═══════════▼═══════════╗            ╔════════▼═════╗ │
        ║  HOOK B                ║           ║  HOOK C      ║ │
        ║  transform the actions ║           ║  bypass      ║ │
        ║  ← 03 action repeat    ║           ║  layers      ║ │
        ╚═══════════╤═══════════╝            ║  ← 04 depth  ║ │
                    │                        ╚══════════════╝ │
              ┌─────▼──────┐                                  │
              │  env.step  │──────────────────────────────────┘
              └────────────┘
```
"""

HOOK_TABLE = """
| hook | what it touches | when it runs | notebook |
|---|---|---|---|
| **A** | the raw camera frame, **before** the policy's own preprocessing | every control step | `02_fixed_foveation` |
| **B** | the action array the policy returned, **before** `env.step` | every control step | `03_action_repeat` |
| **C** | the decoder-layer modules inside the LLM | once per run (calibrate), then in effect for every forward | `04_fixed_depth_pruning` |
"""

WHY_SAME_PLACE = """
### Why the hook points do not change with the backbone

Every one of these methods is defined at a point in the loop that **exists in
every VLA**, not at a point specific to one architecture:

* **Hook A** is defined on the *environment's* frame. Whatever the policy does
  next — SigLIP patches, a VQ tokeniser, whatever — it starts from that frame.
* **Hook B** is defined on the *action array*. Every policy returns one.
* **Hook C** is defined on a `torch.nn.ModuleList` of decoder layers. Every
  LLM-based VLA has one, though it sits at a different attribute path per
  wrapper (`04` walks candidate paths rather than hard-coding one).

That is what makes the comparison meaningful: if backbone A and backbone B are
hooked at different places, a difference in their results says nothing about
the backbones. So when porting to a new benchmark, **keep the hook points and
change only the env/policy adapters.**
"""

PORTING = """
## Porting this to another benchmark (e.g. CALVIN)

**Two** things are benchmark- or backbone-specific. Everything else — the loop,
and all three hooks — stays exactly as it is.

1. **An `EnvAdapter`** for the benchmark: where the camera frame lives, how
   success is reported, whether there is a settle period, and the step cap.
   The gym/gymnasium API difference is already absorbed. CALVIN's success is a
   count of completed subtasks in a sequence rather than a boolean, so decide
   what counts as success for a single episode and put that decision in
   `is_success` — not in the loop.
2. **A policy object** exposing `step(image, instruction) -> (T, action_dim)`
   and `reset()`. A single-action policy may return a flat `(action_dim,)`
   vector; the loop handles both.

One caution when adapting the action convention: gripper sign and
normalisation differ per benchmark **and per checkpoint**, and getting it wrong
produces a policy that reaches correctly but never grasps — which looks like a
method failure rather than a plumbing bug.

### Validate the baseline first

Every result in the grid is a difference against "original policy", so if that
reference is wrong, every other number inherits the error. Before running any
intervention, check the no-intervention condition against the number the
backbone's **own paper** reports for that benchmark.

This is not a formality. Our OpenVLA baseline on `libero_spatial` came out at
74.0% against a published 84.7% — a systematic, reproducible 10.7-point gap
whose cause we still have not identified. Differences measured with the gap
held fixed are still usable, but absolute numbers are not comparable to the
literature until it is understood. Better to find that before the grid than
after.
"""

CAVEAT_MEASURE = """
## Measuring latency without fooling yourself

Two numbers are easy to confuse:

* **ms per model call** — how long one forward costs.
* **ms per environment step** — what the robot actually experiences.

They are only the same when the policy emits one action per call. A policy that
emits a chunk of 10 and executes all of them costs
`model_ms_per_infer / 10` per
environment step. Reporting one as the other makes an already-fast policy look
slow, or vice versa.

Of the three methods here:

| method | ms per call | calls per episode |
|---|---|---|
| foveation | **unchanged** | unchanged |
| action repeat | **unchanged** | **halved** (at repeat=2) |
| depth pruning | **reduced** | unchanged |

So foveation cannot reduce latency at all (the image size, and therefore the
visual token count, is unchanged), action repeat reduces it by making fewer
calls, and depth pruning is the only one that makes a call itself cheaper.
"""


# ------------------------------------------------------------------- 01

cells01 = [
    md("""
    # 01 — Original policy (the reference loop)

    This is the **no-intervention baseline** and the shared skeleton the other
    three notebooks plug into. Read this one first: notebooks 02–04 each
    describe themselves as "hook A/B/C of the loop defined in 01".

    ## What "original policy" means

    It is a *condition*, not a reference architecture. Each backbone runs
    **exactly as its own paper and released checkpoint define it**. Nothing
    about the model is standardised, and no weights are touched anywhere in
    these notebooks.

    What *is* standardised is the loop around the policy:

    | this notebook fixes | the policy still owns |
    |---|---|
    | when the observation is read | image preprocessing (resize, normalise) |
    | the order actions reach `env.step` | vision encoder |
    | how success is decided | LLM architecture and depth |
    | how latency is counted | action tokeniser / de-tokeniser |

    The policy is a black box: it only has to expose
    `step(image, instruction) -> (T, action_dim)` and `reset()`. That is the
    point — if two backbones are scored by different loops, a difference between
    them says nothing about the backbones.
    """),
    md("## The control loop, and the three places a method can attach\n" + LOOP_DIAGRAM),
    md(HOOK_TABLE),
    md(WHY_SAME_PLACE),
    md("""
    ## Everything benchmark-specific lives in one object

    Simulators disagree about almost every detail of the interface, and the
    disagreements are silent or fatal rather than informative:

    | | LIBERO (robosuite) | SimplerEnv (gymnasium) | CALVIN |
    |---|---|---|---|
    | `reset()` returns | `obs` | `(obs, info)` | `obs` |
    | `step()` returns | 4-tuple `(obs, r, done, info)` | **5-tuple** `(obs, r, terminated, truncated, info)` | 4-tuple |
    | success reported via | `done` | `info["success"]` | completed subtasks in a sequence |
    | image lives at | `obs["agentview_image"]` | `obs["image"][cam]["rgb"]` | `obs["rgb_obs"]["rgb_static"]` |
    | settle period at reset | 10 no-op steps | none | none |

    A loop that hard-codes any row of that table is not portable, and unpacking
    a 5-tuple into four names raises `ValueError` the moment it meets gymnasium.

    So all of it goes into an **`EnvAdapter`**. The loop below touches the
    simulator only through that object: porting to a new benchmark means writing
    one adapter — not editing the loop, and never editing the hooks.

    The defaults assume **nothing**: no settle period, success from the
    termination flag or from `info["success"]` when the env provides one.
    LIBERO's conventions are supplied by a LIBERO adapter, as an example rather
    than as the baseline.
    """),
    code('''
    import time
    import numpy as np


    def _unpack_reset(out):
        """gymnasium returns (obs, info); classic gym returns obs."""
        if isinstance(out, tuple) and len(out) == 2 and isinstance(out[1], dict):
            return out[0], out[1]
        return out, {}


    def _unpack_step(out):
        """-> (obs, reward, terminated, truncated, info) for either API."""
        if not isinstance(out, tuple):
            raise TypeError(f"env.step must return a tuple, got {type(out)}")
        if len(out) == 5:
            obs, reward, terminated, truncated, info = out
            return obs, reward, bool(terminated), bool(truncated), info
        if len(out) == 4:
            obs, reward, done, info = out
            return obs, reward, bool(done), False, info
        raise ValueError(
            f"env.step returned {len(out)} values; expected 4 (gym) or 5 "
            f"(gymnasium). Wrap the env or pass a custom adapter.")


    def default_is_success(obs, terminated, info):
        """Prefer an explicit success flag; fall back to termination.

        Stated rather than assumed, because the two disagree: a gymnasium env
        can terminate on a time limit with success False, while LIBERO signals
        success through `done` and puts nothing in info.
        """
        if isinstance(info, dict) and "success" in info:
            return bool(info["success"])
        return bool(terminated)


    def default_grasped(info):
        """Was the source object EVER grasped during this episode?

        ManiSkill2_real2sim keeps this as a cumulative flag in
        `info["episode_stats"]`, so reading the last step's info is enough.
        It separates "never touched the object" from "grasped it and lost
        it", which is what the failure-type analysis is made of, and it is
        independent of whether the episode ultimately succeeded.

        Returns None rather than False when the env does not report it, so
        "not grasped" and "not measured" stay distinguishable in the file.
        """
        if not isinstance(info, dict):
            return None
        stats = info.get("episode_stats")
        if not isinstance(stats, dict) or "is_src_obj_grasped" not in stats:
            return None
        return bool(stats["is_src_obj_grasped"])


    def to_jsonable(value):
        """numpy scalars and arrays -> plain Python.

        An env's info dict is full of numpy bools, and json.dump refuses
        them. Without this the run finishes and then dies on the last line,
        when writing the file -- the most expensive place to fail.
        """
        if isinstance(value, dict):
            return {str(k): to_jsonable(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [to_jsonable(v) for v in value]
        if isinstance(value, np.ndarray):
            return to_jsonable(value.tolist())
        if isinstance(value, np.generic):
            return value.item()
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        return repr(value)


    class EnvAdapter:
        """Wraps a simulator so the loop never sees benchmark-specific details.

        Only `get_image` is mandatory -- there is no defensible default for
        where the camera frame lives, and guessing one would silently read the
        wrong camera rather than fail.
        """

        def __init__(self, env, get_image, is_success=default_is_success,
                     grasped=default_grasped,
                     noop_action=None, settle_steps=0, max_steps=220,
                     owns_env=True):
            self.env = env
            self.get_image = get_image
            self.is_success = is_success
            self.grasped = grasped
            self.noop_action = noop_action
            self.settle_steps = int(settle_steps)
            self.max_steps = int(max_steps)
            # Who closes the simulator. SimplerEnv builds one env per episode,
            # so the adapter owns it and must close it -- a renderer per
            # episode is a leak that ends the run partway through a condition,
            # with no useful error. LIBERO reuses one env across the trials of
            # a task, so there the adapter must NOT close it.
            self.owns_env = bool(owns_env)

            if self.settle_steps and self.noop_action is None:
                raise ValueError("settle_steps > 0 requires a noop_action")

        def reset(self):
            obs, _ = _unpack_reset(self.env.reset())
            return obs

        def step(self, action):
            obs, _, term, trunc, info = _unpack_step(self.env.step(list(action)))
            return obs, term, trunc, info

        def close(self):
            if self.owns_env and hasattr(self.env, "close"):
                self.env.close()
    '''),
    md("""
    ## The policy contract, as we actually implemented it

    All three backbones we ran expose the same two methods, so the loop above
    drives them unchanged. Each wrapper is ~150 lines and does nothing but
    translate between this contract and the checkpoint's own API.

    ```python
    class <Backbone>Inference:
        def reset(self) -> None: ...
        def step(self, image, instruction, wrist_image=None) -> np.ndarray
            # returns (T, action_dim) in the benchmark's action convention
    ```

    What differs between them is only what comes back:

    | backbone | T (actions per call) | views used | notes |
    |---|---|---|---|
    | OpenVLA | **1** | agent only | one action per forward; no chunk exists |
    | UniVLA (Emu3) | ~10 | agent **+ wrist** | raises if the wrist view is missing on a checkpoint trained with it |
    | SpatialVLA | chunk | agent only | accepts `wrist_image` and ignores it |

    Two things belong in the wrapper and nowhere else, because they are
    checkpoint properties rather than method properties:

    * **`unnorm_key`** — which dataset's percentile statistics de-normalise the
      action. Passed explicitly and validated against the keys the checkpoint
      actually ships; a wrong key produces plausible-looking but wrong motion.
    * **the gripper convention** — the training range and sign differ per
      checkpoint. OpenVLA's LIBERO wrapper rescales `[0,1] -> [-1,1]`,
      binarises by sign, then **inverts**, because LIBERO uses `-1 = open`.
      Doing only one of those two steps gives a policy that reaches correctly
      and never grasps.

    Neither belongs in a method notebook — but both must be right before any
    intervention result means anything.
    """),
    md("""
    ## The reference loop

    Deliberately plain. The hooks are `image_fn` / `action_fn` parameters that
    default to identity, so 02 and 03 are one-line changes rather than forks of
    this function.

    The one behaviour worth arguing for: **latching `success`**. A benchmark's
    termination flag usually means "the goal predicate holds *now*". If the arm
    nudges the object afterwards it can flip back, and a solved episode gets
    scored as a failure. Latch it and stop the episode there.
    """),
    code('''
    def identity_image(image, state):
        return image


    def identity_action(actions, state):
        return actions


    def run_episode(adapter, policy, instruction,
                    image_fn=identity_image,      # HOOK A
                    action_fn=identity_action,    # HOOK B
                    state=None):
        """One episode. Returns a dict of per-episode statistics.

        `state` is a free-form dict handed to both hooks, so a hook can keep
        per-episode state (a gaze tracker, a step counter) without this
        function knowing what the hook is.

        The key names below are not cosmetic. They are the names the paired
        tooling reads (`ep_id`, `success`, `steps`, `model_ms_per_infer`,
        `model_ms_per_env_step`), and a record written with other names does
        not load at all rather than loading wrong.
        """
        state = {} if state is None else state
        policy.reset()
        obs = adapter.reset()

        success, done, act_steps = False, False, 0
        model_time, model_calls = 0.0, 0
        info = {}                       # survives an episode with zero steps
        t_start = time.time()

        for _ in range(adapter.settle_steps):      # benchmark-specific; 0 for
            obs, term, trunc, info = adapter.step(adapter.noop_action)
            if term or trunc:                      # most benchmarks
                done = True
                break

        while act_steps < adapter.max_steps and not (success or done):
            image = adapter.get_image(obs)

            # ---- HOOK A: transform the observation ------------------------
            policy_image = image_fn(image, state)

            t0 = time.time()
            actions = policy.step(policy_image, instruction)
            model_time += time.time() - t0
            model_calls += 1

            # ---- HOOK B: transform the actions ----------------------------
            # atleast_2d on both sides: a single-action policy may return a
            # flat (action_dim,) vector, and iterating that would feed the
            # env one scalar per step.
            actions = np.atleast_2d(np.asarray(actions))
            actions = np.atleast_2d(np.asarray(action_fn(actions, state)))

            for row in actions:
                obs, term, trunc, info = adapter.step(row)
                act_steps += 1
                if adapter.is_success(obs, term, info):
                    success = True     # latch: see the note above
                    break
                if term or trunc:
                    done = True
                    break
                if act_steps >= adapter.max_steps:
                    break

        return {
            "success": bool(success),
            "steps": act_steps,
            "elapsed": time.time() - t_start,
            "model_calls": model_calls,
            "model_ms_per_infer": (model_time / model_calls * 1000) if model_calls else 0.0,
            # Amortised over ENV steps, which is what the robot experiences:
            # with a chunk or a repeat of k, one forward covers k env steps.
            "model_ms_per_env_step": (model_time / max(act_steps, 1)) * 1000,
            # env steps executed per observation -- the quantity action repeat
            # actually changes, and not comparable across backbones unless
            # recorded (see 03).
            "steps_per_call": act_steps / model_calls if model_calls else 0.0,
            # Grasp is not success. Keeping it makes "reached but never held"
            # separable from "held it and dropped it" afterwards; not keeping
            # it means re-running the episode to ask.
            "grasped": adapter.grasped(info),
            "episode_stats": to_jsonable(info.get("episode_stats"))
                             if isinstance(info, dict) else None,
        }
    '''),
    md("""
    ## Adapters — the only part that changes per benchmark

    Two examples. Neither is privileged; both are about six lines.

    The settle period and no-op action in the LIBERO adapter come from
    **OpenVLA's own LIBERO evaluation script**
    (`experiments/robot/libero/run_libero_eval.py`). They exist because LIBERO
    drops objects onto the table at reset, so the first steps issue a no-op to
    let the scene settle. **That is a LIBERO fact, not a general one** — which is
    exactly why it lives in an adapter and not in the loop.

    When writing a CALVIN adapter, take these from CALVIN's own reference
    evaluation. The action layout and gripper sign differ between benchmarks,
    and getting the gripper sign wrong yields a policy that reaches correctly
    but never grasps — which looks exactly like a method failure.
    """),
    code('''
    def libero_adapter(env, max_steps=220):
        """LIBERO / robosuite: 4-tuple step, success via `done`, settle first."""
        return EnvAdapter(
            env,
            # The 180-degree flip is part of LIBERO's convention, applied by
            # its reference evaluations; it is not a per-policy choice.
            get_image=lambda obs: obs["agentview_image"][::-1, ::-1],
            is_success=lambda obs, term, info: bool(term),
            noop_action=[0, 0, 0, 0, 0, 0, -1],   # -1 = gripper OPEN in LIBERO
            settle_steps=10,
            max_steps=max_steps,
            # One env serves every trial of a task -- building it is the slow,
            # hang-prone step. Whoever built it closes it.
            owns_env=False,
        )


    def simpler_env_adapter(env, camera="3rd_view_camera", max_steps=120):
        """SimplerEnv / gymnasium: 5-tuple step, success in info, no settle."""
        return EnvAdapter(
            env,
            get_image=lambda obs: obs["image"][camera]["rgb"],
            is_success=lambda obs, term, info: bool(info.get("success", False)),
            settle_steps=0,
            max_steps=max_steps,
        )
    '''),
    md("""
    ## Running a condition

    Every condition must replay the **same initial states**. That is not a
    detail: it turns each comparison into matched pairs instead of two
    independent samples, and a paired test (McNemar) on the same data is far
    more sensitive because episodes where both conditions agree carry no
    information about which is better.

    ### An episode id is not a loop counter

    The number that identifies an episode is the number the **environment**
    uses to fix the initial state, and it has to be passed to the env's
    reset. A bare `for i in range(24)` is not a substitute: it only agrees
    with the env's numbering by accident, and where it disagrees the two
    conditions are scored on different scenes while still looking paired.

    So `run_condition` takes explicit ids per task and refuses to invent
    them. The next section gives the ids for both SimplerEnv suites, where
    the mapping differs by task family.
    """),
    code('''
    import json
    import os


    def run_condition(adapter_factory, policy, tasks, condition="baseline",
                      out_dir=None, model_name="", extra=None, **loop_kwargs):
        """Run one condition over `tasks` and return {task: summary}.

        tasks -- {task_name: [ep_id, ...]}. Explicit, because there is no
                 safe default: on MoveNear the ids are ordered by object
                 triplet, so a prefix such as range(24) is a biased subset
                 rather than a smaller unbiased one.

        adapter_factory(task, ep_id) must reset the env to the initial state
        that env calls `ep_id`, and must do so identically in every
        condition. That is the whole basis of the pairing.

        out_dir -- if given, writes
                   <out_dir>/<condition>/<task>/results_<task>.json,
                   the layout the grid tooling reads.
        extra   -- condition metadata to store at the top of each file, e.g.
                   {"action_repeat": 2} or {"depth_prune": 4}.
        """
        if not isinstance(tasks, dict):
            raise TypeError(
                "tasks must be {task_name: [ep_id, ...]}. Passing a bare task "
                "list would mean guessing the episode ids, and a guess that is "
                "wrong produces two conditions that look paired and are not.")

        out = {}
        for task, ep_ids in tasks.items():
            episodes = []
            for ep_id in ep_ids:
                adapter, instruction = adapter_factory(task, ep_id)
                try:
                    rec = run_episode(adapter, policy, instruction, **loop_kwargs)
                finally:
                    adapter.close()
                rec.update({"task": task, "ep_id": int(ep_id)})
                episodes.append(rec)
                print(f"  {task} ep {ep_id}: "
                      f"{'SUCCESS' if rec['success'] else 'FAIL':<7} "
                      f"{rec['model_ms_per_infer']:.0f} ms/infer", flush=True)

            n_ok = sum(e["success"] for e in episodes)
            n_grasp = sum(1 for e in episodes if e["grasped"])
            summary = {
                "model": model_name,
                "task": task,             # trusted over the directory name
                "condition": condition,
                "n_episodes": len(episodes),
                "success_rate": n_ok / len(episodes) if episodes else 0.0,
                "grasp_rate": n_grasp / len(episodes) if episodes else 0.0,
                "avg_model_ms_per_infer": float(np.mean(
                    [e["model_ms_per_infer"] for e in episodes])),
                "avg_model_ms_per_env_step": float(np.mean(
                    [e["model_ms_per_env_step"] for e in episodes])),
                "avg_steps": float(np.mean([e["steps"] for e in episodes])),
                "avg_steps_per_call": float(np.mean(
                    [e["steps_per_call"] for e in episodes])),
                # Which GPU this ran on. Latency numbers are not comparable
                # across cards, and a file that does not say which one leaves
                # that as a caveat forever.
                "gpu": _gpu_name(),
                "episodes": episodes,   # per-episode records for paired tests
            }
            summary.update(extra or {})
            print(f"[{task}] {n_ok}/{len(episodes)} = "
                  f"{summary['success_rate'] * 100:.1f}%  "
                  f"{summary['avg_model_ms_per_infer']:.0f} ms/infer\\n")

            if out_dir:
                d = os.path.join(out_dir, condition, task)
                os.makedirs(d, exist_ok=True)
                path = os.path.join(d, f"results_{task}.json")
                with open(path, "w") as fh:
                    json.dump(to_jsonable(summary), fh, indent=1)
                print(f"  wrote {path}", flush=True)
            out[task] = summary
        return out


    def _gpu_name():
        try:
            import torch
            if torch.cuda.is_available():
                return torch.cuda.get_device_name(0)
        except Exception:
            pass
        return ""
    '''),
    md("""
    ## The episode ids, for both SimplerEnv suites

    This is the part that cannot be guessed, and the three Google Robot task
    families do it three different ways:

    | task family | how an episode index becomes an initial state |
    |---|---|
    | WidowX-Bridge (4 tasks) | `episode_id`, 0–23 |
    | MoveNear | `episode_id`, 0–59 (ordered by object triplet) |
    | coke can (3 poses) | **no `episode_id` at all** — a 5 × 5 grid of object init xy over `[−0.35, −0.12] × [−0.02, 0.42]`, indexed by the episode number |

    In every case the seed is also set to the episode number. Without it the
    URDF variant is drawn from a per-env RNG that is reseeded whenever the
    env is rebuilt, so two conditions would be scored on different initial
    states — the one failure a paired test cannot detect and cannot survive.

    Coke can is the case that matters most here, because it is the one where
    a `range(...)` counter silently does something else: it does not reach
    the env's placement grid at all.
    """),
    code('''
    # 96 Bridge episodes (4 x 24) and 135 Fractal episodes (3 x 25 + 60).
    EPISODE_IDS = {
        # WidowX-Bridge
        "widowx_put_eggplant_in_basket": list(range(24)),
        "widowx_carrot_on_plate":        list(range(24)),
        "widowx_stack_cube":             list(range(24)),
        "widowx_spoon_on_towel":         list(range(24)),
        # Google Robot / Fractal
        "google_robot_pick_horizontal_coke_can": list(range(25)),
        "google_robot_pick_vertical_coke_can":   list(range(25)),
        "google_robot_pick_standing_coke_can":   list(range(25)),
        "google_robot_move_near_v0":             list(range(60)),
    }

    # "xy_grid" tasks have no episode_id; the index selects a placement.
    XY_GRID_TASKS = {
        "google_robot_pick_horizontal_coke_can",
        "google_robot_pick_vertical_coke_can",
        "google_robot_pick_standing_coke_can",
    }
    XY_GRID = {"x": (-0.35, -0.12, 5), "y": (-0.02, 0.42, 5)}


    def reset_options(task, ep_id):
        """-> (seed, options) turning an episode index into ONE fixed state.

        Kept in one function because every condition has to use the same
        mapping; an episode index that means two different things in two
        conditions is not a paired experiment, and nothing downstream can
        detect that it happened.
        """
        ep_id = int(ep_id)
        if task not in XY_GRID_TASKS:
            return ep_id, {"obj_init_options": {"episode_id": ep_id}}
        xs = np.linspace(*XY_GRID["x"])
        ys = np.linspace(*XY_GRID["y"])
        n = len(xs) * len(ys)
        if ep_id >= n:
            raise ValueError(
                f"episode {ep_id} is outside this task's {len(xs)}x{len(ys)} "
                f"placement grid ({n} distinct initial states). Asking for "
                f"more episodes would re-run the same states and inflate n "
                f"without adding information.")
        x, y = xs[ep_id // len(ys)], ys[ep_id % len(ys)]
        return ep_id, {"obj_init_options": {"init_xy": [float(x), float(y)]}}


    # Sanity check: the two mappings must not be interchangeable.
    assert reset_options("widowx_stack_cube", 7)[1] == {
        "obj_init_options": {"episode_id": 7}}
    assert "init_xy" in reset_options(
        "google_robot_pick_standing_coke_can", 7)[1]["obj_init_options"]
    assert sum(len(v) for k, v in EPISODE_IDS.items() if k.startswith("widowx")) == 96
    assert sum(len(v) for k, v in EPISODE_IDS.items() if k.startswith("google")) == 135
    print("episode-id mapping: 96 Bridge + 135 Fractal, coke can on the xy grid")
    '''),
    md(CAVEAT_MEASURE),
    md("""
    ## Portability check — run this before trusting the loop anywhere

    Two stub environments differing **only** in which simulator API they speak,
    each driven by both a chunked and a single-action policy. All four must
    produce the same outcome.

    This is the check that catches a loop unpacking a gymnasium 5-tuple into
    four names — a `ValueError` on the first step of the first episode, which is
    what the earlier version of this notebook did.
    """),
    code('''
    class _StubBase:
        """Reaches its goal after `solve_at` steps. No physics."""

        def __init__(self, solve_at=40, size=64):
            self.solve_at, self.size, self.t = solve_at, size, 0

        def _obs(self):
            rng = np.random.default_rng(self.t)
            return {"cam": rng.integers(0, 256, (self.size, self.size, 3),
                                        dtype=np.uint8)}


    class ClassicGymEnv(_StubBase):
        """4-tuple step, reset returns obs, success signalled by `done`."""

        def reset(self):
            self.t = 0
            return self._obs()

        def step(self, action):
            self.t += 1
            return self._obs(), 0.0, self.t >= self.solve_at, {}


    class GymnasiumEnv(_StubBase):
        """5-tuple step, reset returns (obs, info), success in info."""

        def reset(self):
            self.t = 0
            return self._obs(), {}

        def step(self, action):
            self.t += 1
            done = self.t >= self.solve_at
            return self._obs(), 0.0, done, False, {"success": done}


    class StubPolicy:
        """Emits a chunk of `chunk` actions per call."""

        def __init__(self, chunk=4, action_dim=7):
            self.chunk, self.action_dim = chunk, action_dim

        def reset(self):
            pass

        def step(self, image, instruction):
            return np.zeros((self.chunk, self.action_dim), dtype=np.float32)


    class SingleActionPolicy(StubPolicy):
        """Emits ONE action per call, as a flat vector.

        The shape most likely to break a loop that assumes 2-D output."""

        def step(self, image, instruction):
            return np.zeros((self.action_dim,), dtype=np.float32)


    get_cam = lambda obs: obs["cam"]

    for env_name, cls in [("classic gym", ClassicGymEnv), ("gymnasium", GymnasiumEnv)]:
        for p_name, pol in [("chunk-4", StubPolicy()), ("single", SingleActionPolicy())]:
            adapter = EnvAdapter(cls(), get_image=get_cam, max_steps=200)
            r = run_episode(adapter, pol, "pick up the black bowl")
            print(f"  {env_name:<12} {p_name:<8} success={r['success']} "
                  f"steps={r['steps']:>3} calls={r['model_calls']:>3} "
                  f"steps/call={r['steps_per_call']:.1f}")
            assert r["success"], f"{env_name}/{p_name} never reached the goal"

    # A settle period must not be counted as policy steps.
    adapter = EnvAdapter(ClassicGymEnv(solve_at=50), get_image=get_cam,
                         noop_action=[0] * 7, settle_steps=10, max_steps=200)
    r = run_episode(adapter, StubPolicy(), "task")
    print(f"\\n  settle_steps=10, goal at env step 50 -> policy steps={r['steps']}")
    assert r["success"] and r["steps"] == 40

    # An env that never succeeds must stop at max_steps, not spin forever.
    adapter = EnvAdapter(ClassicGymEnv(solve_at=10_000), get_image=get_cam,
                         max_steps=60)
    r = run_episode(adapter, StubPolicy(), "task")
    print(f"  unsolvable env -> success={r['success']} steps={r['steps']}")
    assert not r["success"] and r["steps"] == 60

    print("\\nloop is portable across both simulator APIs and both policy shapes")
    '''),
    md("""
    ## The output files, checked by round-tripping them

    A run that finishes and writes a file the analysis cannot read is the
    expensive kind of mistake, so the schema is exercised here on the stub
    env rather than discovered after a real campaign.

    Two conditions are written to a temporary directory and then paired the
    way the analysis pairs them: look up each episode by `ep_id`, keep only
    the pairs whose outcome differs, and count them in both directions. The
    pairing is a dictionary lookup on `ep_id`, which is why that field's
    name and meaning matter more than any other in the file.
    """),
    code('''
    import shutil
    import tempfile


    class _FlakyEnv(GymnasiumEnv):
        """Succeeds only on the episodes listed, so two conditions differ."""

        def __init__(self, solves):
            super().__init__(solve_at=10 if solves else 10_000)


    def _stub_factory(solving_ids):
        def factory(task, ep_id):
            adapter = EnvAdapter(_FlakyEnv(ep_id in solving_ids),
                                 get_image=get_cam, max_steps=30)
            return adapter, "pick up the black bowl"
        return factory


    tmp = tempfile.mkdtemp()
    try:
        plan = {"widowx_stack_cube": list(range(8))}
        run_condition(_stub_factory({0, 1, 2, 3, 4}), StubPolicy(), plan,
                      condition="baseline", out_dir=tmp, model_name="stub")
        run_condition(_stub_factory({0, 1, 5, 6}), StubPolicy(), plan,
                      condition="action_repeat2", out_dir=tmp,
                      model_name="stub", extra={"action_repeat": 2})

        def load(condition, task):
            path = os.path.join(tmp, condition, task, f"results_{task}.json")
            with open(path) as fh:
                summary = json.load(fh)
            # Exactly what the analysis does: index by ep_id, read success.
            return {int(e["ep_id"]): bool(e["success"])
                    for e in summary["episodes"]}

        base = load("baseline", "widowx_stack_cube")
        alt = load("action_repeat2", "widowx_stack_cube")
        shared = sorted(set(base) & set(alt))
        fixed = sum(1 for k in shared if alt[k] and not base[k])
        broke = sum(1 for k in shared if base[k] and not alt[k])

        print(f"\\n  paired on ep_id: {len(shared)} of 8 episodes")
        print(f"  {fixed} fixed / {broke} broke -> "
              f"delta {100 * (fixed - broke) / len(shared):+.1f} points")
        assert len(shared) == 8, "every episode must pair across conditions"
        assert (fixed, broke) == (2, 3)
    finally:
        shutil.rmtree(tmp)

    print("\\nthe written files pair on ep_id and carry the fields the "
          "analysis reads")
    '''),
    md(PORTING),
]
write("01_original_policy.ipynb", cells01)


# ------------------------------------------------------------------- 02

cells02 = [
    md("""
    # 02 — Fixed foveation (HOOK A)

    Foveation imitates the human eye: sharp at the centre of gaze, degraded in
    the periphery. Applied to a robot policy's input it keeps the centre and
    throws away peripheral information.

    **"Fixed"** means the fovea sits at the image centre and stays there. There
    is no tracking, no gaze prediction, no privileged state. That is what makes
    it comparable across backbones and benchmarks — the transform is a pure
    function of the frame.

    The code below is **copied verbatim** from
    `adaptive_sparse_vla/foveation.py`, which is itself a bit-identical port of
    the original RetinaBased OpenVLA implementation (verified by 29 checks
    across 3 image sizes × 4 keep ratios). Copying rather than importing keeps
    this notebook self-contained.
    """),
    md("""
    ## Two variants, and why the difference matters

    | | `blur` | `logpolar` |
    |---|---|---|
    | what it does | progressively **blurs** with distance from the centre | **resamples**: dense at the centre, sparse outward |
    | pixel positions | **unchanged** | **moved** (periphery pulled inward) |
    | what is lost | peripheral sharpness | peripheral spatial resolution |

    `keep_ratio` has the same meaning in both: the effective sampling density
    (log-polar) or the fully-sharp area (blur) is that fraction of the frame.
    `keep_ratio=0.20` is the setting used throughout.

    **This distinction is not cosmetic, and which variant is safe depends on the
    backbone.** Any policy whose visual tokens carry meaning tied to *where* a
    pixel was — a position embedding indexed by patch coordinate, depth
    back-projected through camera intrinsics, an explicit 3D position per token
    — is damaged by a transform that displaces pixels, because after warping
    every token is stamped with the wrong location. A policy that treats the
    image as appearance features without positional grounding is not.

    SpatialVLA is the instance we measured: it back-projects each patch's *grid
    coordinate* through the intrinsics, and log-polar cost it −7.3 points on
    SimplerEnv Bridge while blur, which displaces nothing, recovered most of it.

    **Check this on any new backbone before choosing a variant.** If the model
    computes anything from patch coordinates, `blur` is the honest choice;
    running only log-polar on such a model measures the warp, not foveation.
    """),
    code('''
    import math
    from typing import Optional, Tuple

    import cv2
    import numpy as np


    def _uniform_sample_grid(height, width, keep_ratio):
        keep_ratio = float(np.clip(keep_ratio, 0.0, 1.0))
        if keep_ratio <= 0.0:
            return np.array([0], dtype=np.int32), np.array([0], dtype=np.int32)
        sample_scale = math.sqrt(keep_ratio)
        sample_rows = max(1, int(round(height * sample_scale)))
        sample_cols = max(1, int(round(width * sample_scale)))
        ys = np.linspace(0, height - 1, num=sample_rows, dtype=np.int32)
        xs = np.linspace(0, width - 1, num=sample_cols, dtype=np.int32)
        return ys, xs


    def foveate_image_logpolar(image, keep_ratio, center=None):
        """Warp to log-polar, subsample uniformly there, warp back.

        Sampling uniformly in log-polar space is what produces the radial
        density falloff: equal steps in log-radius are small steps near the
        pole and large ones far from it.
        """
        frame = np.asarray(image, dtype=np.uint8)
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(f"Expected HxWx3 image, got {frame.shape}")
        if keep_ratio <= 0.0:
            return np.zeros_like(frame)

        height, width = frame.shape[:2]
        if center is None:
            center = (width / 2.0, height / 2.0)
        else:
            center = (float(np.clip(center[0], 0, width - 1)),
                      float(np.clip(center[1], 0, height - 1)))
        max_radius = float(np.hypot(max(center[0], width - center[0]),
                                    max(center[1], height - center[1])))
        fwd = cv2.INTER_LINEAR + cv2.WARP_FILL_OUTLIERS + cv2.WARP_POLAR_LOG
        inv = fwd + cv2.WARP_INVERSE_MAP

        logpolar = cv2.warpPolar(frame, (width, height), center, max_radius, fwd)
        ys, xs = _uniform_sample_grid(height, width, keep_ratio)
        sampled = logpolar[np.ix_(ys, xs)]
        interpolated = cv2.resize(sampled, (width, height), interpolation=cv2.INTER_LINEAR)
        restored = cv2.warpPolar(interpolated, (width, height), center, max_radius, inv)
        return np.asarray(np.clip(restored, 0, 255), dtype=np.uint8)


    def foveate_image_blur(image, keep_ratio, center=None):
        """Geometry-preserving foveation: sharp disc, blurred surround, no warp.

        A disc whose area is ~keep_ratio of the frame stays bit-identical;
        outside it, three blur levels are blended with radial weights. Every
        output pixel keeps its input coordinates.
        """
        frame = np.asarray(image, dtype=np.uint8)
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(f"Expected HxWx3 image, got {frame.shape}")

        keep_ratio = float(keep_ratio)
        height, width = frame.shape[:2]
        if keep_ratio >= 1.0:
            return frame.copy()

        if center is None:
            center = (width / 2.0, height / 2.0)
        else:
            center = (float(np.clip(center[0], 0, width - 1)),
                      float(np.clip(center[1], 0, height - 1)))

        r0 = math.sqrt(max(keep_ratio, 0.0) * height * width / math.pi)
        max_radius = float(np.hypot(max(center[0], width - center[0]),
                                    max(center[1], height - center[1])))
        ramp = max(max_radius - r0, 1e-6)

        ys, xs = np.mgrid[0:height, 0:width]
        dist = np.hypot(xs - center[0], ys - center[1])
        t = np.clip((dist - r0) / ramp, 0.0, 1.0).astype(np.float32)

        blur_mid = cv2.GaussianBlur(frame, (0, 0), sigmaX=3.0)
        blur_far = cv2.GaussianBlur(frame, (0, 0), sigmaX=9.0)

        w_far = np.clip(2.0 * t - 1.0, 0.0, 1.0)[..., None]
        w_mid = np.clip(2.0 * t, 0.0, 1.0)[..., None] - w_far
        w_sharp = 1.0 - w_mid - w_far

        out = (frame.astype(np.float32) * w_sharp
               + blur_mid.astype(np.float32) * w_mid
               + blur_far.astype(np.float32) * w_far)
        out = np.clip(np.rint(out), 0, 255).astype(np.uint8)
        out[dist <= r0] = frame[dist <= r0]   # fovea exactly identical
        return out
    '''),
    md("""
    ## HOOK A — where this goes

    On the **raw environment frame, before the policy's own preprocessing.**

    ```
    obs = env.step(...)
    image = get_image(obs)          # (H, W, 3) uint8, e.g. 256×256
    image = foveate_image_blur(image, 0.20)      ← HERE
    actions = policy.step(image, instruction)    # policy resizes to 224 itself
    ```

    ### Getting this wrong is easy and silent

    | placement | consequence |
    |---|---|
    | **before** the policy's resize ✅ | correct: the policy sees a foveated scene at its normal input size |
    | after the resize / inside the model | the transform operates on already-downsampled pixels, so `keep_ratio` no longer means what it says, and results are not comparable to any other backbone |
    | on the normalised float tensor | `cv2` operations on a normalised tensor produce a valid-looking image that is not the intended transform |

    Neither mistake raises an error. Both just lower the success rate, which is
    indistinguishable from "the method does not work".
    """),
    code('''
    def make_foveation_hook(mode="blur", keep_percent=20.0, views=None):
        """Returns an image_fn for `run_episode` in notebook 01.

        Handles whatever shape the adapter's `get_image` returns:

          * a single (H, W, 3) array          -> foveated
          * a dict   {"agent": arr, ...}      -> every entry foveated
          * a list/tuple of arrays            -> every element foveated

        `views` restricts a dict/sequence to named entries. Leaving it None
        degrades EVERY view, which is the condition reported so far -- see the
        note below on why foveating only one view measures the wrong thing.
        """
        fn = foveate_image_blur if mode == "blur" else foveate_image_logpolar
        keep_ratio = keep_percent / 100.0

        def _one(arr):
            return fn(np.asarray(arr), keep_ratio=keep_ratio, center=None)

        def image_fn(image, state):
            if isinstance(image, dict):
                return {k: (_one(v) if (views is None or k in views) else v)
                        for k, v in image.items()}
            if isinstance(image, (list, tuple)):
                return type(image)(
                    _one(v) if (views is None or i in views) else v
                    for i, v in enumerate(image))
            return _one(image)

        return image_fn


    # usage with the loop from 01:
    #   run_episode(adapter, policy, instruction,
    #               image_fn=make_foveation_hook("blur", 20.0))
    '''),
    md("""
    ## How this was wired into the three backbones we ran

    **It was not wired into any of them.** That is the whole point of hook A.

    All three run through one shared evaluation loop, and the transform is
    applied there — on the frame read from the environment, before the frame
    reaches `policy.step`. From `adaptive_sparse_vla/eval_libero.py`:

    ```python
    image = get_libero_image(obs)            # raw env frame, 256x256 uint8
    wrist_image = get_libero_wrist_image(obs)

    policy_image, policy_wrist = image, wrist_image
    if args.foveate:
        policy_image = apply_foveation(image, args, fov_gaze)
        if args.foveate_views == "both":
            policy_wrist = apply_foveation(wrist_image, args, None)

    action_chunk = model.step(policy_image, instruction,
                              wrist_image=policy_wrist)     # <- unchanged
    ```

    `model` here is the OpenVLA, UniVLA or SpatialVLA wrapper, selected by a
    flag. **None of them contains a line of foveation code**, and none of them
    knows whether the frame it received was transformed.

    That is what makes the comparison legitimate: the three backbones get
    pixel-identical inputs from an identical call site, so a difference in
    their results is a difference between the backbones and not between three
    slightly different integrations.

    **To add a fourth backbone, write a wrapper with a `step` method. There is
    no foveation work to do.**
    """),
    md("""
    ## What this assumes about the policy — check before porting

    The transform itself is model-agnostic, but *wiring it in* is not. Four
    assumptions are baked into the hook above; each one is a real difference
    between VLAs, and each fails quietly rather than loudly.

    | assumption | fails when | what to do |
    |---|---|---|
    | **one frame per step** | the policy consumes a *history* of frames (a window of the last 8–16 observations) | foveate every frame entering the window, not just the newest. Foveating one frame in a window of 8 measures a mixed input, and the effective intervention strength is 1/8 of what it says |
    | **uint8 RGB, `H×W×3`** | the env hands back BGR, float, or CHW | convert before, convert back after. `cv2` reads and writes BGR by default; a silent channel swap changes colours the policy was trained on and looks like a method failure |
    | **`keep_ratio` is relative to the frame it is given** | the env renders larger than the policy's input (e.g. 640×480 → 224) | the sharp disc is a fraction of *this* frame, so applying it at 640 then downsampling to 224 is not the same intervention as applying it at 256. Fix the resolution at which foveation happens and record it |
    | **one camera** | the policy reads several views | the hook above already handles a dict or list of views and degrades all of them by default -- but you must still decide and record the choice, see below |

    ### More than one camera

    LIBERO gives an agent view and a wrist view; some setups add more. Decide
    explicitly and record the decision, because "foveate the agent view only"
    and "foveate every view" are different conditions, and the gap between them
    can be as large as the effect being measured.

    Degrading only one view leaves the other as an unfoveated backup, which can
    make a backbone look robust when it is merely reading the other camera. The
    runs reported so far degrade **every view the policy receives**.

    If a gaze-driven centre is ever used, note that a centre computed on the
    agent view is meaningless on a wrist view — they see different scenes. Give
    each view its own centre or leave the secondary views centred.
    """),
    md("""
    ## Visual check

    Always look at the output once before trusting a run. A transform that is
    subtly wrong — fovea in the wrong place, keep ratio misinterpreted,
    channels swapped — produces a plausible image and a quietly lower score.
    """),
    code('''
    def _demo_frame(size=256):
        """Synthetic scene: a few coloured discs on a gradient background."""
        ys, xs = np.mgrid[0:size, 0:size]
        img = np.zeros((size, size, 3), dtype=np.uint8)
        img[..., 0] = (xs * 255 // size).astype(np.uint8)
        img[..., 1] = (ys * 255 // size).astype(np.uint8)
        img[..., 2] = 90
        for (cx, cy, r, col) in [(60, 70, 22, (255, 40, 40)),
                                 (128, 128, 26, (40, 255, 90)),
                                 (200, 190, 20, (60, 120, 255))]:
            m = (xs - cx) ** 2 + (ys - cy) ** 2 <= r * r
            img[m] = col
        # fine texture, so the loss of high-frequency detail is visible
        img[::4, :, :] = np.clip(img[::4, :, :].astype(int) + 45, 0, 255).astype(np.uint8)
        return img


    frame = _demo_frame()
    out_blur = foveate_image_blur(frame, 0.20)
    out_lp = foveate_image_logpolar(frame, 0.20)

    strip = np.concatenate([frame, out_blur, out_lp], axis=1)
    cv2.imwrite("foveation_demo.png", cv2.cvtColor(strip, cv2.COLOR_RGB2BGR))
    print("raw | blur 20% | log-polar 20%  ->  foveation_demo.png")

    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(13, 4.6))
        for i, (title, arr) in enumerate(
                [("raw", frame), ("blur 20%", out_blur), ("log-polar 20%", out_lp)]):
            plt.subplot(1, 3, i + 1); plt.imshow(arr); plt.title(title); plt.axis("off")
        plt.tight_layout(); plt.show()
    except ImportError:
        pass
    '''),
    code('''
    # Properties worth asserting, not eyeballing.
    centre_untouched = np.array_equal(
        foveate_image_blur(frame, 0.20)[120:136, 120:136],
        frame[120:136, 120:136])
    print("blur leaves the fovea bit-identical:", centre_untouched)

    print("blur moves no pixel (shape identical):",
          out_blur.shape == frame.shape)

    def hf_energy(a):
        g = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY).astype(np.float32)
        return float(np.abs(cv2.Laplacian(g, cv2.CV_32F)).mean())

    print(f"high-frequency detail  raw {hf_energy(frame):6.2f}  "
          f"blur {hf_energy(out_blur):6.2f}  logpolar {hf_energy(out_lp):6.2f}")
    print("\\n-> both remove detail; only log-polar also displaces pixels.")
    '''),
    md("""
    ## The hook survives every observation shape

    `get_image` returns whatever the benchmark's adapter gives it: one array on
    a single-camera setup, a dict on LIBERO (agent + wrist), possibly a list.
    The hook must not care -- otherwise it works on one benchmark and silently
    skips views on another.
    """),
    code('''
    single = _demo_frame(128)
    multi_dict = {"agent": _demo_frame(128), "wrist": _demo_frame(96)}
    multi_list = [_demo_frame(128), _demo_frame(96)]

    hook_all = make_foveation_hook("blur", 20.0)

    out = hook_all(single, {})
    assert isinstance(out, np.ndarray) and out.shape == single.shape
    print("single array  ->", out.shape)

    out = hook_all(multi_dict, {})
    assert set(out) == {"agent", "wrist"}
    assert all(not np.array_equal(out[k], multi_dict[k]) for k in out)
    print("dict of views ->", {k: v.shape for k, v in out.items()},
          "| both degraded")

    out = hook_all(multi_list, {})
    assert len(out) == 2 and all(not np.array_equal(a, b)
                                 for a, b in zip(out, multi_list))
    print("list of views ->", [v.shape for v in out], "| both degraded")

    # Restricting to named views must leave the others bit-identical -- that is
    # the "agent only" condition, and it is a DIFFERENT experiment.
    hook_agent = make_foveation_hook("blur", 20.0, views={"agent"})
    out = hook_agent(multi_dict, {})
    assert not np.array_equal(out["agent"], multi_dict["agent"])
    assert np.array_equal(out["wrist"], multi_dict["wrist"])
    print("views={\'agent\'}  -> wrist left untouched (a different condition)")

    print("\\nhook is shape-agnostic")
    '''),
    md("""
    ## Two properties of the method itself

    * **It cannot reduce latency.** The output has the same dimensions as the
      input, so the policy sees the same number of visual tokens and does the
      same work. This follows from the transform, not from any measurement:
      foveation is an accuracy intervention. If a speedup is wanted it has to
      come from somewhere else (notebook 04).
    * **The sign of the effect is not fixed.** Removing peripheral detail can
      delete clutter or delete signal depending on where the task-relevant
      content sits in the frame. Nothing about the transform guarantees which,
      so treat the direction as an empirical question per scene — not a
      property to be assumed from another benchmark.
    """),
    md("""
    ## Appendix — what we observed on our own runs

    Context only. **None of this is a property of the method** — it is what
    happened on the backbones and benchmarks we ran, at 50–96 episodes per
    condition depending on the benchmark. At those sizes a difference of
    roughly 10 points or less is not reliably distinguishable from chance.
    Do not carry these numbers to a new setup — carry the questions.

    | | observation |
    |---|---|
    | latency | unchanged in every condition we ran (two backbones, ±6 ms) |
    | direction, benchmark A | log-polar 20% moved one backbone clearly up (+18.8) and two others by less than the resolution (+8.3, −7.3) |
    | direction, benchmark B | the same code hurt both of the backbones that had gone up on A, one of them decisively (−74) |
    | fovea placement | placing the fovea on the target using simulator ground truth — an upper bound no deployable gaze can beat — did not recover the loss (58% → 50%, not distinguishable from chance) |

    The third row is the one worth repeating on a new setup, because it is a
    cheap way to find out whether a foveation loss is about *where* the fovea
    is or about *how much* was removed.
    """),
]
write("02_fixed_foveation.ipynb", cells02)


# ------------------------------------------------------------------- 03

cells03 = [
    md("""
    # 03 — Action repeat (HOOK B)

    The temporal intervention: call the model **half as often** and hold each
    action for two environment steps instead of one.

    ```python
    actions = np.repeat(actions, 2, axis=0)
    ```

    That single line is the whole mechanism. What follows is why it goes where
    it goes, how it differs from chunk execution, and how to measure it without
    reporting the wrong number.
    """),
    md("""
    ## Action repeat vs chunk execution — these are different interventions

    Both reduce the number of model calls. They are not interchangeable.

    | | action repeat 2 | chunk-exec 2 |
    |---|---|---|
    | what gets executed | the **same action, copied** | **two different actions** the model actually predicted |
    | requires | nothing | the policy must emit multiple actions per call |
    | information lost | yes — motion becomes stepwise | comparatively little |
    | works on a single-action policy | ✅ | ❌ **not defined** |

    **A policy that emits one action per call has no chunk to truncate**, so
    chunk-exec is not a worse option for it — it does not exist. Action repeat
    is therefore the only temporal intervention that runs *identically*
    regardless of whether a policy chunks, which is what makes it comparable
    across a set of backbones that do not all chunk.

    That is itself a finding worth stating in the analysis: the temporal axis
    is really two mechanisms — one universal but lossy, one lossless but
    requiring native action chunking — and a backbone that cannot chunk is
    forced onto the worse one.

    In our own runs the two came out in opposite directions on different
    backbones — chunk-exec at k=2 was one backbone's best result (+13.6 at 1.9×
    faster) and cost another 12.5 points. That is context, not a prediction:
    see the appendix at the end.
    """),
    md("""
    ## HOOK B — where this goes

    On the **action array the policy returned, before it reaches `env.step`.**

    ```
    actions = policy.step(image, instruction)   # (T, action_dim)
    actions = np.repeat(actions, 2, axis=0)     ← HERE   -> (2T, action_dim)
    for row in actions:
        env.step(row)
    ```

    ### Order matters if combined with chunk truncation

    If a run uses both, truncate **first**, then repeat:

    ```python
    actions = actions[:k]                       # chunk-exec
    actions = np.repeat(actions, r, axis=0)     # action repeat
    ```

    Repeating first and then truncating silently produces a different
    condition — with `k=2, r=2` it would execute the first action twice and
    nothing else, rather than two actions twice each.
    """),
    code('''
    import numpy as np


    def apply_action_repeat(actions, repeat):
        """Hold each action for `repeat` consecutive environment steps.

        np.repeat (not np.tile): repeat=2 on [a, b, c] gives [a, a, b, b, c, c],
        whereas tile would give [a, b, c, a, b, c] — a completely different
        trajectory that would still run and still produce a number.
        """
        actions = np.asarray(actions)
        if repeat <= 1:
            return actions
        return np.repeat(actions, int(repeat), axis=0)


    def make_action_repeat_hook(repeat=2, exec_chunk=0):
        """Returns an action_fn for `run_episode` in notebook 01."""

        def action_fn(actions, state):
            if exec_chunk > 0:
                actions = actions[:exec_chunk]      # truncate first
            return apply_action_repeat(actions, repeat)

        return action_fn


    # usage with the loop from 01:
    #   run_episode(adapter, policy, instruction,
    #               action_fn=make_action_repeat_hook(repeat=2))
    '''),
    code('''
    demo = np.array([[0.1, 0.0], [0.2, 0.0], [0.3, 0.0]])
    print("original      :", demo[:, 0].tolist())
    print("repeat 2      :", apply_action_repeat(demo, 2)[:, 0].tolist())
    print("tile (WRONG)  :", np.tile(demo, (2, 1))[:, 0].tolist())
    '''),
    md("""
    ## How this was wired into the three backbones we ran

    Also nowhere — same as hook A. One line in the shared loop, after every
    policy has already returned. From `adaptive_sparse_vla/eval_libero.py`:

    ```python
    action_chunk = model.step(policy_image, instruction, wrist_image=policy_wrist)

    if args.exec_chunk > 0:
        action_chunk = action_chunk[: args.exec_chunk]        # truncate first
    if args.action_repeat > 1:
        action_chunk = np.repeat(action_chunk, args.action_repeat, axis=0)

    for action_row in action_chunk:
        obs, _, done, _ = env.step(action_row.tolist())
    ```

    It works unchanged whether `action_chunk` came back with one row (OpenVLA)
    or ten (UniVLA), which is exactly why this is the temporal condition that
    is comparable across all of them.

    **To add a fourth backbone: nothing to do.**
    """),
    md("""
    ## ⚠️ Before running this on a new benchmark: is the action space relative?

    > **Who this is for:** whoever sets up a benchmark this method has not been
    > run on yet. **When:** once, at setup — not per run.
    >
    > Already checked: **LIBERO** (robosuite `OSC_POSE`) and **SimplerEnv** are
    > both relative, so results from them are interpretable as-is.
    > **Not checked: CALVIN**, which exposes *both* relative and absolute action
    > modes, with the active one depending on the config and on how the policy
    > was trained.

    Repeating an action does something completely different depending on what
    an action *is*:

    | action space | an action means | repeating it twice |
    |---|---|---|
    | **relative / delta** (`Δx, Δy, Δz, Δrot, gripper`) | "move 5 cm forward" | moves 10 cm — the arm travels **twice as far** open-loop. The intervention as intended. |
    | **absolute** (target joint angles or end-effector pose) | "go to position (30, 20)" | already there — the second step is a **no-op** and the arm holds still. |

    ### Why this is worth ten minutes

    In an absolute action space the condition **costs nothing and does nothing**:

    * success rate barely moves — no real intervention was applied
    * model calls genuinely halve — the model really was called half as often

    The results table then reads **"2× faster at no cost in accuracy"**, which
    looks like the strongest result in the whole study and is entirely an
    artifact of the controller absorbing the repeat. Nothing about the run
    errors, warns, or looks wrong.

    ### The check

    Hold one action for several steps and watch whether the arm keeps moving.
    No theory needed.

    ```python
    obs = env.reset()
    a = policy.step(get_image(obs), instruction)[0]   # one action

    positions = []
    for _ in range(10):
        obs, *_ = env.step(list(a))                  # the SAME action, 10 times
        positions.append(read_ee_position(obs))      # however the env exposes it

    print(positions)
    # keeps drifting  -> relative   run this condition; results are meaningful
    # stops after one -> absolute   do NOT run it; repeat is a no-op here
    ```

    If the action space turns out to be absolute, the temporal axis needs a
    different intervention on that benchmark — not this one.

    ## What it costs the trajectory (relative action spaces)

    A repeated action doubles the displacement commanded before the policy sees
    a new frame, so the arm travels twice as far open-loop between corrections.
    The failure mode is therefore *overshoot on approach and imprecision at
    contact*, not a uniform degradation — which is why it hurts tasks needing
    fine placement far more than coarse reaching.
    """),
    code('''
    # A policy tracking a target with proportional control, under repeat=1 vs 2.
    # The gain is stable when each action is applied once; repeating it doubles
    # the effective gain, which is what pushes the loop past 1.0 and overshoots.
    def simulate(repeat, steps=40, gain=0.6, target=1.0):
        pos, trace = 0.0, []
        t = 0
        while t < steps:
            action = gain * (target - pos)          # one model call
            for _ in range(repeat):                 # executed `repeat` times
                pos += action
                trace.append(pos)
                t += 1
                if t >= steps:
                    break
        return trace

    a, b = simulate(1), simulate(2)
    print(f"repeat=1  final {a[-1]:.3f}   max overshoot {max(a) - 1.0:+.3f}")
    print(f"repeat=2  final {b[-1]:.3f}   max overshoot {max(b) - 1.0:+.3f}")
    print("\\nBoth converge here because the target does not move. On a real "
          "task\\nthe overshoot lands the gripper past the object, and the "
          "correction\\narrives one full call late.")

    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(7, 3.2))
        plt.axhline(1.0, color="gray", ls="--", lw=1, label="target")
        plt.plot(a, label="repeat=1")
        plt.plot(b, label="repeat=2")
        plt.xlabel("environment step"); plt.ylabel("position")
        plt.legend(); plt.tight_layout(); plt.show()
    except ImportError:
        pass
    '''),
    md("""
    ## Measuring it correctly

    **The cost of one model call does not change.** What halves is the number
    of calls. So the per-call figure is identical to baseline and reporting it
    alone would suggest the intervention did nothing.

    Report either:

    * **calls per episode** (halved), or
    * **ms per environment step** = `model_ms_per_infer × calls / env_steps` (halved),

    and say which. The `run_episode` in notebook 01 returns both
    `model_ms_per_infer` and `model_ms_per_env_step` for exactly this reason.

    One more asymmetry, and it is structural rather than empirical: a policy
    that already emits a chunk of 10 and executes all of them is *already* amortised
    10× per call. Applying repeat=2 on top pushes it to 20 environment steps of
    open-loop execution between observations, while the same flag on a
    single-action policy pushes it to 2.

    **So repeat=2 is not one intervention strength across backbones.** The
    quantity that actually determines the damage is *environment steps executed
    per observation*, and repeat multiplies whatever the policy already does:

    | | steps/obs at baseline | at repeat=2 |
    |---|---|---|
    | single-action policy | 1 | 2 |
    | chunk-10 policy | 10 | **20** |

    If the goal is to compare backbones at matched open-loop horizon rather
    than at a matched flag value, record steps-per-observation alongside the
    success rate — otherwise a backbone can look fragile when it was simply
    pushed twice as far.
    """),
    md("""
    ## Check the hook does what it claims

    Self-contained: a stub env and a stub policy, no simulator. What it pins
    down is that repeat halves the **calls** while leaving the number of
    environment steps and the per-call cost alone — the exact confusion the
    section above warns about.
    """),
    code('''
    class _CountingEnv:
        """Counts env steps; never terminates, so the step cap decides."""

        def __init__(self, size=32):
            self.size, self.n = size, 0

        def reset(self):
            self.n = 0
            return {"cam": np.zeros((self.size, self.size, 3), np.uint8)}

        def step(self, action):
            self.n += 1
            return {"cam": np.zeros((self.size, self.size, 3), np.uint8)}, 0.0, False, {}


    class _ChunkPolicy:
        def __init__(self, chunk):
            self.chunk = chunk

        def reset(self):
            pass

        def step(self, image, instruction):
            return np.zeros((self.chunk, 7), dtype=np.float32)


    def _drive(chunk, repeat, max_steps=80):
        """Minimal driver, so this cell does not depend on notebook 01."""
        env, pol = _CountingEnv(), _ChunkPolicy(chunk)
        hook = make_action_repeat_hook(repeat=repeat)
        obs, steps, calls = env.reset(), 0, 0
        while steps < max_steps:
            actions = hook(np.asarray(pol.step(obs["cam"], "task")), {})
            calls += 1
            for row in actions:
                obs, *_ = env.step(row)
                steps += 1
                if steps >= max_steps:
                    break
        return steps, calls


    print(f"{'policy':<12}{'repeat':>7}{'env steps':>11}{'calls':>7}{'steps/call':>12}")
    for chunk in (1, 10):
        base_steps, base_calls = _drive(chunk, 1)
        rep_steps, rep_calls = _drive(chunk, 2)
        for label, (st, ca) in [("baseline", (base_steps, base_calls)),
                                ("repeat 2", (rep_steps, rep_calls))]:
            print(f"chunk-{chunk:<7}{label:>7}{st:>11}{ca:>7}{st / ca:>12.1f}")
        assert rep_calls * 2 == base_calls, "repeat=2 must halve the calls"
        assert rep_steps == base_steps, "env steps must be unchanged"

    print("\\ncalls halve; env steps do not. The per-call cost is untouched,")
    print("so reporting ms/call alone would show no effect at all.")
    '''),
    md("""
    ## Appendix — what we observed on our own runs

    Context only. **None of this is a property of the method** — it is what
    happened on the backbones and benchmarks we ran, at 50–96 episodes per
    condition depending on the benchmark. At those sizes a difference of
    roughly 10 points or less is not reliably distinguishable from chance.
    Do not carry these numbers to a new setup — carry the questions.

    | condition | observation |
    |---|---|
    | repeat 2, single-action policy | −8 points, not distinguishable from chance |
    | repeat 2, chunk-10 policy | **−68 points** |
    | chunk-exec k=2, one chunking backbone | **+13.6 points at 1.9× faster** |
    | chunk-exec k=2, another chunking backbone | −12.5 points |

    Rows 1 and 2 are the same flag at very different open-loop horizons (2 vs
    20 env steps per observation), so they are not two measurements of one
    intervention strength. Rows 3 and 4 are the same intervention with opposite
    signs on two backbones that both support it.

    The transferable lesson is the bookkeeping, not the numbers: **record
    env-steps-per-observation next to every temporal result**, or a backbone
    can look fragile when it was simply pushed ten times further.
    """),
]
write("03_action_repeat.ipynb", cells03)


# ------------------------------------------------------------------- 04

cells04 = [
    md("""
    # 04 — Fixed depth pruning (HOOK C)

    Skip **N decoder layers** of the LLM entirely: no computation, hidden
    states pass straight through to the next layer. No retraining, no
    fine-tuning, no change to the weights that remain.

    **This is the only one of the three that makes a model call cheaper**, and
    the size of the saving follows one quantity: **the share of the step spent
    inside the decoder stack.** Removing N of L layers removes roughly N/L of
    that share, and nothing else. So the first thing to do on any backbone is
    profile a control step and find out what that share is — everything the
    method can possibly buy is bounded by it.

    For scale, one backbone we profiled spent 6% on visual encoding, 13% on
    prefill and **70% on autoregressive decode**, which is why attacking the
    visual path there was capped at ~19% however aggressive it got. **That
    split is a property of that model, not a constant** (see the assumptions
    section below for architectures where it does not hold at all).

    **"Fixed"** means the same N layers stay bypassed for the whole episode.
    (A phase-adaptive variant that changes N mid-episode exists and is
    deliberately not in this notebook — it is a separate condition.)
    """),
    md("""
    ## The method in three steps

    1. **Measure**, once, how much each layer changes the representation:
       `1 − cos(layer_input, layer_output)`. Small means the layer barely moves
       the hidden state — it is idle.
    2. **Rank** and pick the N most idle, subject to two safeguards (below).
    3. **Replace** those modules with a pass-through.

    Step 1 rides on the first inference of the run, which has to happen
    anyway, so calibration costs **no extra forward pass**. It happens once
    per run, not once per episode: the bypassed set is a property of the
    condition being measured.

    ### The two safeguards, and why each exists

    | safeguard | rule | why |
    |---|---|---|
    | **protect the early stack** | only the back half is eligible | Early layers perform the foundational transforms everything downstream depends on. On one backbone we ran, bypassing layers 2 and 4 made generation never terminate at all. |
    | **enforce a gap** | no two bypassed layers adjacent | Consecutive removals compound — the second layer's input is already wrong — so a gap-respecting greedy pass runs first, then the remainder fills in. |

    Both thresholds (`min_layer=0.5`, `min_gap=1`) are **heuristics tuned on the
    backbones we ran**, not derived quantities. They are exposed as arguments
    rather than hard-coded so a new backbone can be swept. What must not change
    between backbones is the *rule*: the claim "backbone A tolerates more depth
    removal than backbone B" only means something if both were ranked and cut
    identically.

    ### Two conventions exist, and they must not be mixed inside one table

    Our own runs did not all use the settings above. SpatialVLA's stack was
    cut under a different convention, and `rank_layers` takes both:

    | | OpenVLA, UniVLA | SpatialVLA |
    |---|---|---|
    | how `min_layer` reads | `window="ratio"` — 0.5 = the back half | `window="count"` — 2 = skip layers 0–1 |
    | eligible range on a 26-layer stack | L13–25 | L2–25 |
    | gap rule | `min_gap=1` | `min_gap=0` (none) |
    | final layer | eligible | always protected |

    The same argument value means different things under the two readings,
    which is why the reading is named explicitly rather than guessed from
    the number. Pick one per backbone family and keep it fixed; a column cut
    under one convention and a column cut under the other differ in more
    than the layer count.
    """),
    md("""
    ## The pass-through layer

    The subtle part is not skipping the computation; it is the **KV cache**.

    `transformers`' `DynamicCache` indexes by `layer_idx`. A layer that never
    calls `cache.update()` leaves a gap in that list, and a *later* layer's
    update then raises `IndexError: list index out of range`. So the bypass
    still writes a correctly-shaped zero placeholder. It is never read — this
    layer has no attention — so zeros are safe.

    Getting this wrong does not always crash. It can instead silently corrupt
    the cache, which shows up only as a lower success rate — indistinguishable
    from "the method does not work". The check at the end of this notebook
    exists for that reason.
    """),
    code('''
    from typing import Any, Optional

    import torch


    class BypassDecoderLayer(torch.nn.Module):
        """Skips the layer's compute, writes a placeholder KV to keep the cache
        contiguous."""

        def __init__(self, layer_idx: int, num_kv_heads: Optional[int] = None,
                     head_dim: Optional[int] = None):
            super().__init__()
            self.layer_idx = int(layer_idx)
            self.num_kv_heads = num_kv_heads
            self.head_dim = head_dim

        def forward(self, hidden_states, attention_mask=None, position_ids=None,
                    past_key_value=None, output_attentions=False, use_cache=False,
                    **kwargs):
            if (use_cache and past_key_value is not None
                    and hasattr(past_key_value, "update")
                    and self.num_kv_heads is not None and self.head_dim is not None):
                bsz, seq_len = hidden_states.shape[0], hidden_states.shape[1]
                dummy = torch.zeros(bsz, self.num_kv_heads, seq_len, self.head_dim,
                                    dtype=hidden_states.dtype,
                                    device=hidden_states.device)
                past_key_value.update(dummy, dummy, self.layer_idx, {})
            outputs = (hidden_states,)
            if output_attentions:
                outputs += (None,)
            if use_cache:
                outputs += (past_key_value,)
            return outputs
    '''),
    md("""
    ## HOOK C — where this goes

    Not in the control loop: **inside the model**, once per episode.

    ```
    policy.reset()
    ├─ first policy.step(...) of the episode
    │    └─ forward hooks record 1 − cos(in, out) per layer   ← MEASURE
    ├─ rank, pick N, swap those modules for BypassDecoderLayer ← APPLY
    └─ every subsequent step runs the pruned stack
    ```

    Finding the layer stack is the only backbone-specific part, and it is
    handled by walking candidate attribute paths rather than hard-coding one.
    A model that *is* the language model exposes the stack at `model.layers`;
    one wrapped by a multimodal head buries it a level or two deeper, at
    `model.language_model.model.layers` or similar. A hard-coded path that
    silently misses is how an "unpruned" run gets reported as pruned, so
    `find_decoder_layers` returns `None` and the caller refuses to run rather
    than guessing.
    """),
    code('''
    import math
    import numpy as np


    def find_decoder_layers(model):
        """Locate the decoder stack across the wrappers different VLAs use.

        Wrappers nest the language model at different depths, so the attribute
        path is not portable -- one exposes it directly, another buries it
        under a multimodal wrapper. Walking candidates beats hard-coding a
        path, which is how a silently-unpruned run gets reported as pruned.
        """
        candidates = (
            ("model", "layers"),
            ("language_model", "model", "layers"),
            ("language_model", "layers"),
            ("model", "language_model", "layers"),
            ("layers",),
        )
        for path in candidates:
            node = model
            for attr in path:
                node = getattr(node, attr, None)
                if node is None:
                    break
            if isinstance(node, torch.nn.ModuleList) and len(node) > 0:
                return node
        return None


    def measure_redundancy_with_hooks(layers, run_fn):
        """Per-layer 1 - cos(in, out), captured while run_fn() executes.

        Only the FIRST call of each layer is recorded. Generation calls every
        layer once per decoded token; the prefill -- the first call -- is the
        one that sees the whole prompt. Averaging in single-token decode steps
        would measure something else.

        Must run with the stack UNPRUNED: a bypassed layer has input == output,
        so it would report ~0 redundancy and rank itself most-redundant forever.
        """
        scores = [None] * len(layers)
        handles = []

        def make_hook(idx):
            def hook(module, args, kwargs, output):
                if scores[idx] is not None:
                    return
                inp = args[0] if args else kwargs.get("hidden_states")
                out = output[0] if isinstance(output, tuple) else output
                if inp is None or out is None or not torch.is_tensor(inp):
                    return
                cos = torch.nn.functional.cosine_similarity(
                    inp.float(), out.float(), dim=-1)
                scores[idx] = float(1.0 - cos.mean().item())
            return hook

        for i, layer in enumerate(layers):
            handles.append(layer.register_forward_hook(make_hook(i), with_kwargs=True))
        try:
            run_fn()
        finally:
            for h in handles:
                h.remove()
        return None if any(s is None for s in scores) else [float(s) for s in scores]


    def rank_layers(importance, min_layer=0.5, min_gap=1, window="ratio",
                    protect_last=False):
        """Eligible layers, most-redundant first, gap-respecting.

        The defaults are the OpenVLA / UniVLA convention. SpatialVLA's is
        different in all three respects, and the two are not interchangeable:

            OpenVLA, UniVLA:  window="ratio", min_layer=0.5, min_gap=1,
                              protect_last=False
            SpatialVLA:       window="count", min_layer=2,   min_gap=0,
                              protect_last=True

        `window` says how to read `min_layer`. Under "ratio", 0.5 means "the
        back half"; under "count", 2 means "skip layers 0 and 1". On a
        26-layer stack those start at layer 13 and layer 2 -- the same
        argument naming two very different experiments -- so the reading is
        spelled out rather than inferred from the value.

        `min_gap=0` disables the gap rule (no two layers are ever within 0
        of each other), which is what SpatialVLA's selection does.
        """
        scores = np.asarray(importance, dtype=np.float32)
        n = int(scores.shape[0])
        if window == "ratio":
            start = int(math.floor(np.clip(min_layer, 0.0, 1.0) * n))
        elif window == "count":
            start = int(np.clip(int(min_layer), 0, n))
        else:
            raise ValueError(f"window must be 'ratio' or 'count', got {window!r}")
        candidates = [i for i in range(start, n)
                      if not (protect_last and i == n - 1)]
        if not candidates:
            candidates = list(range(n))
        ranked = sorted(candidates, key=lambda i: (float(scores[i]), i))
        ordered = []
        for i in ranked:                       # gap-respecting greedy first
            if any(abs(i - prev) <= min_gap for prev in ordered):
                continue
            ordered.append(i)
        for i in ranked:                       # then fill in the rest
            if i not in ordered:
                ordered.append(i)
        return ordered
    '''),
    code('''
    class StaticDepthPruner:
        """Calibrate once per RUN, then bypass the N most redundant layers.

        Per run, not per episode. Every number in the grid was produced this
        way: the ranking is measured on the first forward of the first
        episode and then held fixed for the whole condition, so the bypassed
        set is a property of the condition rather than of an episode.

        Re-calibrating each episode is a perfectly defensible experiment --
        it is just a different one, and mixing the two inside one table
        would compare conditions that differ in more than the layer count.
        It is available as `recalibrate_each_episode=True`, opt-in.
        """

        def __init__(self, model, prune=8, min_layer=0.5, min_gap=1,
                     window="ratio", protect_last=False,
                     recalibrate_each_episode=False):
            self.model, self.prune = model, int(prune)
            self.min_layer, self.min_gap = min_layer, min_gap
            self.window, self.protect_last = window, bool(protect_last)
            self.recalibrate_each_episode = bool(recalibrate_each_episode)
            self._originals, self._active, self._done = {}, (), False

        def layers(self):
            return find_decoder_layers(self.model)

        def restore(self):
            layers = self.layers()
            if layers is None or not self._originals:
                return
            for idx, layer in self._originals.items():
                layers[idx] = layer
            self._originals, self._active = {}, ()

        def _head_shape(self):
            """Read the attention shape, following nested configs.

            Multimodal wrappers usually keep the language model's config as a
            sub-config; the outer one carries no attention shape at all, and a
            wrong head_dim produces a KV placeholder the cache cannot
            concatenate.
            """
            cfg = getattr(self.model, "config", None)
            for attr in ("text_config", "llm_config", "language_model_config"):
                sub = getattr(cfg, attr, None)
                if sub is not None and getattr(sub, "num_attention_heads", None):
                    cfg = sub
                    break
            n_heads = getattr(cfg, "num_attention_heads", None)
            hidden = getattr(cfg, "hidden_size", None)
            kv = getattr(cfg, "num_key_value_heads", None) or n_heads
            head_dim = getattr(cfg, "head_dim", None) or (
                (hidden // n_heads) if (hidden and n_heads) else None)
            if kv is None or head_dim is None:
                # Falling through with None would make BypassDecoderLayer skip
                # the cache write entirely, leaving a gap that surfaces either
                # as an IndexError deep in a later layer or as quiet cache
                # corruption. Refuse instead: an unpruned run reported as
                # pruned is the worst outcome available here.
                raise RuntimeError(
                    "could not determine (num_kv_heads, head_dim) from this "
                    "model's config, so the KV placeholder would be the wrong "
                    "shape. Pass them explicitly after reading the model's own "
                    "attention implementation.")
            return kv, head_dim

        def apply(self, indices):
            layers = self.layers()
            if layers is None:
                return
            valid = tuple(i for i in sorted({int(x) for x in indices})
                          if 0 <= i < len(layers))
            self.restore()          # restore first, or bypasses accumulate
            if not valid:
                return
            kv, head_dim = self._head_shape()
            for idx in valid:
                self._originals[idx] = layers[idx]
                layers[idx] = BypassDecoderLayer(idx, num_kv_heads=kv,
                                                 head_dim=head_dim)
            self._active = valid

        def calibrate_on(self, run_fn):
            """Call with the model's real first forward. Rides on it, so it
            costs no extra pass; after the first call it is a no-op."""
            if self._done:
                return run_fn()
            layers = self.layers()
            if layers is None:
                raise RuntimeError(
                    "decoder stack not found -- refusing to report an unpruned "
                    "run as pruned. Add this model's attribute path to "
                    "find_decoder_layers.")
            captured = {}
            importance = measure_redundancy_with_hooks(
                layers, lambda: captured.setdefault("out", run_fn()))
            if importance is not None:
                ranking = rank_layers(importance, self.min_layer, self.min_gap,
                                      window=self.window,
                                      protect_last=self.protect_last)
                self.apply(ranking[:self.prune])
                print(f"[depth] bypassing {list(self._active)} of {len(layers)}")
            self._done = True
            return captured.get("out")

        def reset_episode(self):
            """Call between episodes. A no-op unless recalibration is on.

            Deliberately not "clear the ranking": with the default the
            bypassed set has to stay fixed across the condition, and a
            reset here would quietly turn a run into per-episode
            recalibration -- a different experiment that looks the same in
            the output file.
            """
            if not self.recalibrate_each_episode:
                return
            self._done = False
            self.restore()
    '''),
    md("""
    ## How this was wired into the three backbones we ran

    This is the one hook that **does** need per-backbone code — and it is worth
    being precise about which part, because the rest is shared on purpose.

    ### Shared, identical for every backbone (`adaptive_sparse_vla/depth_prune.py`)

    Locating the stack, the redundancy metric, the ranking rule, both
    safeguards, the bypass module, and the restore bookkeeping. A claim that
    backbones differ in exploitable depth redundancy only means something if
    every backbone was ranked and cut by the same rule, so this lives in one
    file that all three import.

    ### Per-backbone: only *how the measurement is taken*

    The metric is the same — layer input vs layer output on the real prompt's
    prefill — but the backbones do not expose the same call path.

    **UniVLA (Emu3)** can be called directly, so it asks for hidden states:

    ```python
    out = self.model.model(input_ids=..., attention_mask=...,
                           use_cache=False, output_hidden_states=True,
                           return_dict=True)
    hs = out.hidden_states
    return [1.0 - cos(a, b).mean() for a, b in zip(hs[:-1], hs[1:])]
    ```

    Costs **one extra prefill** per calibration, which slightly inflates this
    backbone's measured ms/infer — i.e. it biases *against* depth pruning here,
    not for it.

    **OpenVLA** cannot: its `generate` is wrapped inside `predict_action`, with
    no way to ask for hidden states. So forward hooks ride along on the first
    real action prediction of the episode:

    ```python
    if self.depth.needs_calibration():
        captured = {}
        importance = measure_redundancy_with_hooks(
            self.depth.layers(), lambda: captured.setdefault("a", _predict()))
        action = captured.get("a")          # reuse the hooked call's output
        self.depth.calibrate(importance)
    else:
        action = _predict()
    ```

    Costs **no extra forward pass** — that episode's first step simply runs
    unpruned, 1 of up to ~230.

    **SpatialVLA (Gemma2)** needed a separate implementation
    (`SpatialVLA/experiments/tome/depth_prune_gemma2.py`) because its cache is
    a pre-allocated sliding-window `HybridCache` rather than the per-layer list
    the shared bypass writes into.

    ### What this means for a fourth backbone

    Two questions, in order:

    1. Does `find_decoder_layers` return the stack? If not, add its path — do
       not let it return `None` and run anyway.
    2. Does cached generation still match uncached greedy decoding once layers
       are bypassed? If not, the cache is not a per-layer list and the
       placeholder needs rewriting for that cache type.

    Everything else — ranking, safeguards, the N-layer choice — is already
    shared and should not be re-implemented.
    """),
    md("""
    ## What this assumes about the architecture — check before porting

    This is the most architecture-dependent of the three methods. Foveation
    touches an image and action repeat touches an array; this one reaches inside
    the model, so it inherits assumptions the other two do not.

    | assumption | fails when | symptom / what to do |
    |---|---|---|
    | **the stack is a flat `ModuleList` of interchangeable decoder layers** | the model interleaves a different layer type — e.g. gated cross-attention blocks in Flamingo-style architectures | bypassing a cross-attention block removes the *vision pathway*, not redundant compute. Inspect the module list and exclude any layer type that is not a plain self-attention block |
    | **the decoder dominates the step** | the LLM runs **once** per step and a small head (MLP, LSTM, diffusion/flow) produces the action | there is no autoregressive decode to shrink, so the saving is only on prefill and is much smaller. Profile first; the intervention may not be worth running |
    | **cache is a per-layer list indexed by `layer_idx`** (`DynamicCache`) | the model uses a static or hybrid cache with pre-allocated shape, e.g. Gemma2's sliding-window `HybridCache` | the zero placeholder may not be the shape the cache expects. Verify cached vs uncached generation produce identical tokens *on that model* |
    | **KV shape is `(batch, kv_heads, seq, head_dim)`** | attention variants that do not store K and V in that layout (e.g. latent-compressed attention) | the placeholder cannot be concatenated. Read the model's own attention code before trusting `_head_shape()` |
    | **bypassing preserves sequence semantics** | layers carry per-layer positional or rotary state that later layers depend on | rare, but shows up as coherent-looking output that ignores the instruction |

    None of these produce a clean error. Each produces a lower success rate,
    which is indistinguishable from "depth pruning does not work on this
    backbone" — the exact claim the experiment is trying to test. That is why
    the verification cell below is not optional.

    ## Wiring it into a policy

    ```python
    class Policy:
        def __init__(self, model, prune=8):
            self.model = model
            self.depth = StaticDepthPruner(model, prune=prune)

        def reset(self):
            self.depth.reset_episode()      # re-measure on the unpruned stack

        def step(self, image, instruction):
            inputs = self.preprocess(image, instruction)
            run = lambda: self.model.generate(**inputs)
            out = self.depth.calibrate_on(run)     # measures on call 1, then pruned
            return self.detokenise(out)
    ```

    One episode therefore runs its first step unpruned — 1 of up to ~230 — and
    every step after that on the pruned stack.
    """),
    md("""
    ## Check: does a pruned model with cache still decode correctly?

    This is the check that matters. If the KV placeholder is wrong, `generate()`
    with a cache diverges from uncached greedy decoding, and the only symptom is
    a lower success rate weeks later.

    The cell below builds a small stack in plain PyTorch and verifies the bypass
    behaves as a true identity on hidden states while keeping the cache
    contiguous. On a real checkpoint, run the same comparison with
    `model.generate(..., use_cache=True)` against `use_cache=False`; they must
    produce **identical token ids**, not merely similar ones.
    """),
    code('''
    class _FakeLayer(torch.nn.Module):
        def __init__(self, idx, hidden=16, kv_heads=2, head_dim=8):
            super().__init__()
            self.idx, self.kv_heads, self.head_dim = idx, kv_heads, head_dim
            self.lin = torch.nn.Linear(hidden, hidden)
            # Layer 3 and 5 are near-identity: they should rank most redundant.
            with torch.no_grad():
                if idx in (3, 5):
                    self.lin.weight.copy_(torch.eye(hidden) + 1e-4)
                    self.lin.bias.zero_()

        def forward(self, hidden_states, past_key_value=None, use_cache=False, **kw):
            out = hidden_states + 0.02 * self.lin(hidden_states)
            if use_cache and past_key_value is not None:
                b, s = hidden_states.shape[0], hidden_states.shape[1]
                d = torch.zeros(b, self.kv_heads, s, self.head_dim)
                past_key_value.update(d, d, self.idx, {})
            return (out, past_key_value) if use_cache else (out,)


    class _FakeCache:
        def __init__(self):
            self.slots = []
        def update(self, k, v, idx, _):
            while len(self.slots) <= idx:
                self.slots.append(None)
            if self.slots[idx] is not None:
                raise IndexError("layer wrote twice")
            self.slots[idx] = (k, v)
            return k, v


    class _FakeModel(torch.nn.Module):
        def __init__(self, n=8, hidden=16):
            super().__init__()
            self.layers = torch.nn.ModuleList([_FakeLayer(i, hidden) for i in range(n)])
            class _Cfg: num_attention_heads = 2; hidden_size = hidden; num_key_value_heads = 2
            self.config = _Cfg()
        def forward(self, x, use_cache=False):
            cache = _FakeCache() if use_cache else None
            for layer in self.layers:
                x = layer(x, past_key_value=cache, use_cache=use_cache)[0]
            return x, cache


    torch.manual_seed(0)
    model = _FakeModel()
    x = torch.randn(1, 5, 16)

    print("[1] the stack is found:", find_decoder_layers(model) is not None)

    imp = measure_redundancy_with_hooks(model.layers, lambda: model(x))
    print("[2] per-layer 1-cos(in,out):", [round(v, 5) for v in imp])

    order = rank_layers(imp, min_layer=0.5, min_gap=1)
    print("[3] ranked (most redundant first, back half only):", order)
    assert min(order[:2]) >= 4, "early layers must never be eligible"

    pruner = StaticDepthPruner(model, prune=2)
    pruner.calibrate_on(lambda: model(x))
    bypassed = pruner._active
    print("[4] bypassed:", list(bypassed))

    # A bypassed layer must be an exact identity on hidden states...
    layer = model.layers[bypassed[0]]
    probe = torch.randn(1, 3, 16)
    assert torch.equal(layer(probe)[0], probe)
    print("[5] bypassed layer is an exact identity on hidden states")

    # ...and the cache must stay contiguous with no gaps.
    _, cache = model(x, use_cache=True)
    print("[6] cache slots filled:", sum(s is not None for s in cache.slots),
          "of", len(model.layers))
    assert all(s is not None for s in cache.slots), "gap in the KV cache"

    pruner.restore()
    assert not isinstance(model.layers[bypassed[0]], BypassDecoderLayer)
    print("[7] restore() puts the real modules back")

    # The two conventions must actually differ, and the SpatialVLA one must
    # never hand back the final layer.
    n = len(model.layers)
    ratio_order = rank_layers(imp, min_layer=0.5, min_gap=1, window="ratio")
    count_order = rank_layers(imp, min_layer=2, min_gap=0, window="count",
                              protect_last=True)
    assert min(ratio_order) >= n // 2
    assert min(count_order) >= 2 and (n - 1) not in count_order
    assert count_order == sorted(count_order, key=lambda i: (imp[i], i))
    print(f"[8] ratio window {ratio_order} vs count window {count_order} "
          f"(last layer {n - 1} protected)")

    # Default: calibration survives reset_episode, so the bypassed set is
    # fixed for the whole run.
    p = StaticDepthPruner(model, prune=2)
    p.calibrate_on(lambda: model(x))
    fixed = p._active
    p.reset_episode()
    assert p._active == fixed, "default must not re-calibrate between episodes"
    p.restore()

    p = StaticDepthPruner(model, prune=2, recalibrate_each_episode=True)
    p.calibrate_on(lambda: model(x))
    p.reset_episode()
    assert p._active == (), "opt-in recalibration must clear the bypass"
    p.restore()
    print("[9] calibration is once per run unless recalibration is opted into")
    print("\\nALL CHECKS PASSED")
    '''),
    md("""
    ## Three rules that hold regardless of backbone

    * **Calibrate on the unpruned stack.** A bypassed layer has
      input == output, so measuring it while already bypassed scores it ~0
      redundancy and locks it in permanently. This is also why calibration
      is once per run: a second pass would be taken on an already-pruned
      stack unless the layers are restored first.
    * **Do not carry a value of N between backbones — measure the curve.** How
      much depth a model can spare is a property of that model, and the rule
      above only ranks layers; it does not tell you how many are safe to cut.
    * **Verify the speedup actually happened.** Bypassing N of L layers should
      reduce ms/call by roughly N/L. If success dropped and latency did not
      move, the layers were not really bypassed — check that
      `find_decoder_layers` found the right stack rather than concluding the
      method failed.
    """),
    md("""
    ## Appendix — what we observed on our own runs

    Context only. **None of this is a property of the method** — it is what
    happened on the backbones and benchmarks we ran, at 50–96 episodes per
    condition depending on the benchmark. At those sizes a difference of
    roughly 10 points or less is not reliably distinguishable from chance.
    Do not carry these numbers to a new setup — carry the questions.

    Ranked and cut by the **identical rule**, how much depth each backbone
    could spare differed enormously:

    | backbone | layers | bypassed | result |
    |---|---|---|---|
    | A | 32 | 8 (25%) | −10 points |
    | B | 32 | 8 (25%) | **−46 points** |
    | C | 26 | **1** (4%) | lost accuracy on 3 of 4 tasks |

    A and B are directly comparable — same depth, same ratio, and a 4.6× gap in
    what it cost. C is not on that scale at all: it was never run at 25%,
    because a single bypassed layer already hurt. Read C as "broke immediately",
    not as a third point on the same curve.

    The spread is the finding. It is also the reason the selection rule lives in
    one shared place: a claim that backbones differ in exploitable depth
    redundancy means nothing unless every backbone was ranked and cut
    identically.
    """),
]
write("04_fixed_depth_pruning.ipynb", cells04)


# ------------------------------------------------------------------- 05

cells05 = [
    md("""
    # 05 — Running the campaign

    `01`–`04` define the loop and the three interventions and prove them on
    stubs. They deliberately name no model and no benchmark. This notebook is
    the other half: it builds the real environments, sweeps the eight
    conditions, and writes the files the analysis reads.

    **You supply one thing: a policy object.** Everything else is here.

    | | where it comes from |
    |---|---|
    | the control loop, the file format, the episode ids | `01` |
    | foveation, action repeat, depth pruning | `02`, `03`, `04` |
    | SimplerEnv and LIBERO environments | this notebook |
    | the eight conditions and the driver | this notebook |
    | **the policy** | **you** |

    Run order in a fresh session: put all five notebooks in one directory and
    start here — the next cell loads `01`–`04` for you.
    """),
    code('''
    # Load 01-04 into this session. They are read from the .ipynb files rather
    # than copied, so this notebook cannot drift from them.
    import json
    import os

    NOTEBOOK_DIR = os.getcwd()
    _NEEDED = ["01_original_policy", "02_fixed_foveation",
               "03_action_repeat", "04_fixed_depth_pruning"]

    for _name in _NEEDED:
        _path = os.path.join(NOTEBOOK_DIR, _name + ".ipynb")
        if not os.path.exists(_path):
            raise FileNotFoundError(
                f"{_path} not found. Put all five notebooks in one directory, "
                f"or set NOTEBOOK_DIR to where they live. Copying the code in "
                f"by hand instead is how the notebook and the runs drift "
                f"apart.")
        with open(_path) as fh:
            _nb = json.load(fh)
        print(f"--- {_name} ---", flush=True)
        for _cell in _nb["cells"]:
            if _cell["cell_type"] == "code":
                exec(compile("".join(_cell["source"]), _name, "exec"), globals())

    # Their self-checks just ran, so if this line prints, the loop, the two
    # image transforms, the action transform and the bypass layer all work in
    # this environment -- before any checkpoint is downloaded.
    for _fn in ["run_episode", "run_condition", "EPISODE_IDS", "reset_options",
                "make_foveation_hook", "make_action_repeat_hook",
                "StaticDepthPruner", "find_decoder_layers"]:
        assert _fn in globals(), f"{_fn} missing -- did a cell above fail?"
    print("\\n01-04 loaded and self-checked")
    '''),
    md("""
    ## The one thing you supply

    ```python
    class YourPolicy:
        def reset(self) -> None: ...
        def step(self, image, instruction) -> np.ndarray   # (T, action_dim)
    ```

    A single-action policy may return a flat `(action_dim,)` vector; the loop
    handles both. For depth pruning the object also needs a `.model` attribute
    holding the `torch.nn.Module` whose decoder stack should be pruned.

    Two things live in this wrapper and nowhere else, and both fail **quietly**
    — they lower the success rate instead of raising:

    * **`unnorm_key`** — which dataset's percentile statistics de-normalise the
      action. Pass it explicitly and check it against the keys the checkpoint
      actually ships. A wrong key produces plausible-looking but wrong motion.
    * **the gripper convention** — training range and sign differ per
      checkpoint. Get it wrong and the arm reaches correctly and never grasps,
      which looks exactly like a method failure.

    `check_policy` below catches the shape mistakes. It cannot catch either of
    those two; only the baseline-vs-paper comparison at the end can.
    """),
    code('''
    def check_policy(policy, image_shape=(256, 256, 3), instruction="pick up the cup"):
        """Fail now, on one fake frame, rather than 90 episodes in."""
        for name in ("reset", "step"):
            if not callable(getattr(policy, name, None)):
                raise AttributeError(f"policy has no callable {name}()")
        policy.reset()
        frame = np.zeros(image_shape, dtype=np.uint8)
        actions = np.atleast_2d(np.asarray(policy.step(frame, instruction)))
        if actions.ndim != 2:
            raise ValueError(
                f"policy.step returned shape {actions.shape}; expected "
                f"(T, action_dim) or a flat (action_dim,)")
        chunk, dim = actions.shape
        has_model = isinstance(getattr(policy, "model", None), torch.nn.Module)
        print(f"  returns {chunk} action(s) of dim {dim}")
        print(f"  .model present: {has_model}"
              f"{'' if has_model else '  <- depth pruning needs this'}")
        if has_model:
            layers = find_decoder_layers(policy.model)
            if layers is None:
                print("  decoder stack: NOT FOUND -- add the attribute path to "
                      "find_decoder_layers in 04 before running depth prune")
            else:
                print(f"  decoder stack: {len(layers)} layers")
        # Native chunk length decides how action repeat composes: a policy
        # already emitting 10 actions per call is amortised 10x before the
        # intervention, so repeat=2 lands at 20 env steps per forward.
        print(f"  native chunk length: {chunk}"
              f"{'  (repeat composes on top of this)' if chunk > 1 else ''}")
        return {"chunk": chunk, "action_dim": dim, "has_model": has_model}
    '''),
    md("""
    ## SimplerEnv environments

    Two families with different construction paths, and neither detail is
    cosmetic.

    **Bridge** needs the `rgb_overlay_path` real-inpainting image. The
    checkpoints were evaluated against that composite; without it the policy is
    scored on a distribution it has never seen. It does not raise — it just
    scores badly — so the builder below refuses to run instead.

    **Fractal** needs `prepackaged_config=True`, which sets robot, control
    mode, frequencies, scene and overlay from the env itself. Re-deriving those
    by hand is how a run silently loses its overlay.

    `SIMPLER_ROOT` must point at your SimplerEnv checkout.
    """),
    code('''
    SIMPLER_ROOT = "/content/SimplerEnv"    # <- set this

    BRIDGE_TASKS = {
        "widowx_put_eggplant_in_basket": {
            "env_name": "PutEggplantInBasketScene-v0",
            "robot": "widowx_sink_camera_setup",
            "scene_name": "bridge_table_1_v2",
            "rgb_overlay_path": "ManiSkill2_real2sim/data/real_inpainting/bridge_sink.png",
            "max_episode_steps": 120,
        },
        "widowx_carrot_on_plate": {
            "env_name": "PutCarrotOnPlateInScene-v0",
            "robot": "widowx",
            "scene_name": "bridge_table_1_v1",
            "rgb_overlay_path": "ManiSkill2_real2sim/data/real_inpainting/bridge_real_eval_1.png",
            "max_episode_steps": 60,
        },
        "widowx_stack_cube": {
            "env_name": "StackGreenCubeOnYellowCubeBakedTexInScene-v0",
            "robot": "widowx",
            "scene_name": "bridge_table_1_v1",
            "rgb_overlay_path": "ManiSkill2_real2sim/data/real_inpainting/bridge_real_eval_1.png",
            "max_episode_steps": 60,
        },
        "widowx_spoon_on_towel": {
            "env_name": "PutSpoonOnTableClothInScene-v0",
            "robot": "widowx",
            "scene_name": "bridge_table_1_v1",
            "rgb_overlay_path": "ManiSkill2_real2sim/data/real_inpainting/bridge_real_eval_1.png",
            "max_episode_steps": 60,
        },
    }

    FRACTAL_TASKS = {
        "google_robot_pick_horizontal_coke_can": {
            "env_name": "GraspSingleOpenedCokeCanInScene-v0",
            "env_kwargs": {"lr_switch": True}, "max_episode_steps": 80},
        "google_robot_pick_vertical_coke_can": {
            "env_name": "GraspSingleOpenedCokeCanInScene-v0",
            "env_kwargs": {"laid_vertically": True}, "max_episode_steps": 80},
        "google_robot_pick_standing_coke_can": {
            "env_name": "GraspSingleOpenedCokeCanInScene-v0",
            "env_kwargs": {"upright": True}, "max_episode_steps": 80},
        "google_robot_move_near_v0": {
            "env_name": "MoveNearGoogleBakedTexInScene-v0",
            "env_kwargs": {}, "max_episode_steps": 80},
    }

    # Bridge renders the policy view from 3rd_view_camera, Fractal from
    # overhead_camera. Reading the wrong one is silent.
    CAMERA = {**{t: "3rd_view_camera" for t in BRIDGE_TASKS},
              **{t: "overhead_camera" for t in FRACTAL_TASKS}}


    def build_bridge_env(task, ep_id):
        from simpler_env.utils.env.env_builder import (
            build_maniskill2_env, get_robot_control_mode)

        cfg = BRIDGE_TASKS[task]
        overlay = None
        for base in (SIMPLER_ROOT, os.path.join(SIMPLER_ROOT, "ManiSkill2_real2sim")):
            cand = os.path.join(base, cfg["rgb_overlay_path"])
            if os.path.exists(cand):
                overlay = cand
                break
        if overlay is None:
            raise FileNotFoundError(
                f"visual-matching overlay not found: {cfg['rgb_overlay_path']} "
                f"(searched under {SIMPLER_ROOT}). Refusing to run: without it "
                f"the policy is scored on the raw sim render, which is not the "
                f"distribution it was evaluated on, and nothing about the run "
                f"will look wrong.")
        env = build_maniskill2_env(
            cfg["env_name"], obs_mode="rgbd", robot=cfg["robot"],
            sim_freq=500, control_freq=5,
            control_mode=get_robot_control_mode(cfg["robot"], "policy"),
            max_episode_steps=cfg["max_episode_steps"],
            scene_name=cfg["scene_name"],
            camera_cfgs={"add_segmentation": True},
            renderer_kwargs={"offscreen_only": True},
            rgb_overlay_path=overlay,
            rgb_overlay_cameras=["3rd_view_camera"],
        )
        _, options = reset_options(task, ep_id)          # from 01
        obs, _ = env.reset(seed=int(ep_id), options=options)
        return env, obs


    def build_fractal_env(task, ep_id):
        import gymnasium as gym
        import simpler_env  # noqa: F401 -- registers the env ids

        cfg = FRACTAL_TASKS[task]
        env = gym.make(cfg["env_name"], obs_mode="rgbd", prepackaged_config=True,
                       max_episode_steps=cfg["max_episode_steps"],
                       **cfg["env_kwargs"])
        _, options = reset_options(task, ep_id)          # from 01
        obs, _ = env.reset(seed=int(ep_id), options=options)
        if not getattr(env.unwrapped, "rgb_overlay_path", None):
            raise RuntimeError(
                f"{task}: SimplerEnv returned no rgb_overlay_path. The "
                f"real_inpainting assets are missing under "
                f"{SIMPLER_ROOT}/ManiSkill2_real2sim/data. Refusing to run "
                f"rather than evaluate on the raw sim render.")
        return env, obs


    def simpler_adapter_factory(task, ep_id):
        """The (task, ep_id) -> (adapter, instruction) callable run_condition wants."""
        build = build_bridge_env if task in BRIDGE_TASKS else build_fractal_env
        env, _ = build(task, ep_id)
        instruction = env.get_language_instruction()
        cfg = (BRIDGE_TASKS if task in BRIDGE_TASKS else FRACTAL_TASKS)[task]
        cam = CAMERA[task]

        def get_image(obs):
            from simpler_env.utils.env.observation_utils import (
                get_image_from_maniskill2_obs_dict)
            return get_image_from_maniskill2_obs_dict(env, obs, camera_name=cam)

        adapter = EnvAdapter(
            env, get_image=get_image,
            is_success=lambda obs, term, info: bool(info.get("success", False)),
            settle_steps=0, max_steps=cfg["max_episode_steps"])
        return adapter, instruction


    BRIDGE_PLAN = {t: EPISODE_IDS[t] for t in BRIDGE_TASKS}       # 96 episodes
    FRACTAL_PLAN = {t: EPISODE_IDS[t] for t in FRACTAL_TASKS}     # 135 episodes
    print("Bridge ", sum(len(v) for v in BRIDGE_PLAN.values()), "episodes")
    print("Fractal", sum(len(v) for v in FRACTAL_PLAN.values()), "episodes")
    '''),
    md("""
    ## LIBERO environments

    Three differences from SimplerEnv, all of which have bitten us:

    1. **LIBERO cannot be run zero-shot.** Success collapses to roughly 0
       without a LIBERO fine-tune, and each released checkpoint is trained on
       **one suite only**. Always pair the checkpoint with its own suite;
       crossing them measures nothing.
    2. **The initial states are an explicit array.**
       `env.set_init_state(init_states[i])` fully determines the episode, so
       here the pairing key really is `(task_id, trial)` — a plain counter,
       unlike the SimplerEnv coke-can tasks.
    3. **Rendering.** On current Colab images EGL hangs or segfaults once a
       policy is resident on the GPU. Set `MUJOCO_GL=osmesa` **before** the
       first LIBERO import; OSMesa renders on the CPU and cannot contend with
       CUDA.

    One env per task, reset per trial — the same convention as OpenVLA's
    official eval. Building an env is the slow, hang-prone step; rebuilding it
    per trial multiplies that by the trial count for no benefit.
    """),
    code('''
    os.environ.setdefault("MUJOCO_GL", "osmesa")     # before any libero import

    LIBERO_SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]
    LIBERO_RESOLUTION = 256
    LIBERO_MAX_STEPS = 220


    def libero_plan(suite, n_tasks=10, n_trials_per_task=5):
        """-> {task_name: [trial, ...]}, one entry per LIBERO task.

        The task name carries the suite so the written files stay separable
        when several suites share an output directory.
        """
        return {f"{suite}_task{tid:02d}": list(range(n_trials_per_task))
                for tid in range(n_tasks)}


    class _LiberoEpisode:
        """One fixed initial state, presented as a plain env.

        `reset()` has to do two things on LIBERO -- reset, then apply the
        stored init state -- and the loop only calls one. Wrapping keeps that
        detail out of EnvAdapter.
        """

        def __init__(self, env, init_state):
            self.env, self.init_state = env, init_state

        def reset(self):
            self.env.reset()
            return self.env.set_init_state(self.init_state)

        def step(self, action):
            return self.env.step(action)


    class LiberoTasks:
        """Suite loader that keeps one env alive per task."""

        def __init__(self):
            self._suites, self._env, self._env_key = {}, None, None

        def suite(self, name):
            if name not in self._suites:
                from libero.libero import benchmark
                self._suites[name] = benchmark.get_benchmark_dict()[name]()
            return self._suites[name]

        def _build(self, suite_name, task_id):
            from libero.libero import get_libero_path
            from libero.libero.envs import OffScreenRenderEnv

            task = self.suite(suite_name).get_task(task_id)
            bddl = os.path.join(get_libero_path("bddl_files"),
                                task.problem_folder, task.bddl_file)
            env = OffScreenRenderEnv(bddl_file_name=bddl,
                                     camera_heights=LIBERO_RESOLUTION,
                                     camera_widths=LIBERO_RESOLUTION)
            env.seed(0)
            return env, task

        def __call__(self, task_name, trial):
            """The (task, ep_id) -> (adapter, instruction) callable, so this
            object can be passed to run_campaign directly and closed by it."""
            suite_name, _, tail = task_name.rpartition("_task")
            task_id = int(tail)
            key = (suite_name, task_id)
            if self._env_key != key:
                self.close()
                self._env, self._task = self._build(suite_name, task_id)
                self._env_key = key
            states = self.suite(suite_name).get_task_init_states(task_id)
            episode = _LiberoEpisode(self._env, states[trial % len(states)])
            # owns_env=False inside: this object closes the env, not the loop.
            adapter = libero_adapter(episode, max_steps=LIBERO_MAX_STEPS)  # from 01
            return adapter, self._task.language

        def close(self):
            if self._env is not None:
                self._env.close()
            self._env, self._env_key = None, None


    # Usage:
    #   run_campaign(policy, LiberoTasks(), libero_plan("libero_spatial"), out)
    # run_campaign closes it for you when it finishes.
    print("LIBERO suites:", ", ".join(LIBERO_SUITES))
    print("per suite:", sum(len(v) for v in libero_plan("libero_spatial").values()),
          "episodes per condition")
    '''),
    md("""
    ## The eight conditions

    Six of them are one argument. Depth pruning is not, and the difference is
    structural rather than incidental: foveation and action repeat are
    functions applied *around* the policy, while depth pruning replaces modules
    *inside* it.

    | condition | how |
    |---|---|
    | `baseline` | no hooks |
    | `action_repeat2` / `action_repeat4` | `action_fn=make_action_repeat_hook(repeat=k)` |
    | `foveate_logpolar` | `image_fn=make_foveation_hook("logpolar", 20.0)` |
    | `foveate_blur` | `image_fn=make_foveation_hook("blur", 20.0)` |
    | `depth_prune1` / `2` / `4` | wrap `policy.step` so the first call calibrates |

    The condition names are the directory names the analysis expects. Keep
    them.

    **`DEPTH_KW` is the layer-window convention, and it is per backbone
    family** — see `04`. The default is OpenVLA/UniVLA's; SpatialVLA's is
    `window="count", min_layer=2, min_gap=0, protect_last=True`. Pick one per
    model and keep it fixed across all its columns.
    """),
    code('''
    CONDITIONS = [
        ("baseline",         {}),
        ("action_repeat2",   {"repeat": 2}),
        ("action_repeat4",   {"repeat": 4}),
        ("foveate_logpolar", {"foveate": ("logpolar", 20.0)}),
        ("foveate_blur",     {"foveate": ("blur", 20.0)}),
        ("depth_prune1",     {"prune": 1}),
        ("depth_prune2",     {"prune": 2}),
        ("depth_prune4",     {"prune": 4}),
    ]

    DEPTH_KW = dict(window="ratio", min_layer=0.5, min_gap=1, protect_last=False)


    def attach_depth_pruning(policy, prune, **kw):
        """Wrap policy.step so the first call calibrates and the rest run pruned.

        This is the OpenVLA shape from 04: the hooks ride on the first real
        action prediction, so calibration costs no extra forward pass. It
        works whenever policy.step() performs the model forward.

        If your policy's forward happens where these hooks cannot see it --
        a generate() buried inside a helper, or a cache type the bypass does
        not write into -- attach the pruner at that call instead. 04's
        "How this was wired" section has the three shapes we hit.
        """
        model = getattr(policy, "model", None)
        if not isinstance(model, torch.nn.Module):
            raise AttributeError(
                "depth pruning needs policy.model to be the torch module whose "
                "decoder stack should be pruned. Expose it, or attach the "
                "pruner inside the wrapper.")
        if find_decoder_layers(model) is None:
            raise RuntimeError(
                "decoder stack not found -- refusing to report an unpruned run "
                "as pruned. Add this model's attribute path to "
                "find_decoder_layers in 04.")
        pruner = StaticDepthPruner(model, prune=prune, **kw)
        original = policy.step

        def step(image, instruction, *a, **kwargs):
            return pruner.calibrate_on(
                lambda: original(image, instruction, *a, **kwargs))

        policy.step = step
        policy._depth = (pruner, original)
        return pruner


    def detach_depth_pruning(policy):
        """Put the real layers and the real step back. Always call this."""
        state = getattr(policy, "_depth", None)
        if state is None:
            return
        pruner, original = state
        pruner.restore()
        policy.step = original
        del policy._depth
    '''),
    md("""
    ## The driver

    One call per cell of the table: one model, one benchmark, eight conditions.

    Order matters in one respect only — `baseline` runs first, so a setup
    problem shows up before seven intervention runs have been spent on it.
    Otherwise the conditions are independent; `skip` lets you resume after an
    interruption without redoing what is already on disk.
    """),
    code('''
    def run_campaign(policy, adapter_factory, plan, out_dir, model_name="",
                     conditions=None, skip=()):
        """Run all eight conditions over `plan` and write them under `out_dir`.

        Returns {condition: {task: summary}}. Files land at
        <out_dir>/<condition>/<task>/results_<task>.json.
        """
        conditions = list(conditions or CONDITIONS)
        done = {}
        try:
            _run_conditions(policy, adapter_factory, plan, out_dir, model_name,
                            conditions, skip, done)
        finally:
            # LiberoTasks holds one env open across a whole condition; nothing
            # else in the loop can know when it is finished with.
            closer = getattr(adapter_factory, "close", None)
            if callable(closer):
                closer()

        print("\\n" + "=" * 64)
        for name, per_task in done.items():
            n = sum(s["n_episodes"] for s in per_task.values())
            ok = sum(s["success_rate"] * s["n_episodes"] for s in per_task.values())
            print(f"  {name:<18} {ok:>5.0f}/{n:<5} = {100 * ok / n:5.1f}%")
        return done


    def _run_conditions(policy, adapter_factory, plan, out_dir, model_name,
                        conditions, skip, done):
        for name, spec in conditions:
            if name in skip:
                print(f"[skip] {name}", flush=True)
                continue
            print(f"\\n{'=' * 64}\\n  {name}\\n{'=' * 64}", flush=True)
            kwargs, extra = {}, {}
            if "repeat" in spec:
                kwargs["action_fn"] = make_action_repeat_hook(repeat=spec["repeat"])
                extra["action_repeat"] = spec["repeat"]
            if "foveate" in spec:
                mode, keep = spec["foveate"]
                kwargs["image_fn"] = make_foveation_hook(mode, keep)
                extra["foveate"] = {"mode": mode, "keep_percent": keep}
            if "prune" in spec:
                attach_depth_pruning(policy, spec["prune"], **DEPTH_KW)
                extra["depth_prune"] = spec["prune"]
                extra["depth_window"] = dict(DEPTH_KW)
            try:
                done[name] = run_condition(
                    adapter_factory, policy, plan, condition=name,
                    out_dir=out_dir, model_name=model_name, extra=extra,
                    **kwargs)
            finally:
                # In a finally block on purpose: an exception mid-condition
                # would otherwise leave the next one running on a pruned
                # stack and silently report it as something else.
                detach_depth_pruning(policy)


    # One cell of the table:
    #   run_campaign(policy, simpler_adapter_factory, BRIDGE_PLAN,
    #                "results/mymodel_bridge", model_name="MyVLA")
    '''),
    md("""
    ## Dry run — the whole driver, no simulator and no checkpoint

    Everything above except the two `build_*_env` functions runs here on stubs.
    What it proves: the eight conditions produce eight directories with the
    expected names, the depth conditions really bypass layers and really put
    them back, and the written files pair on `ep_id`.

    Run this before downloading anything.
    """),
    code('''
    import shutil
    import tempfile


    class _StubVLA:
        """Policy shaped like a real one: a torch model and a chunk of actions.

        `_FakeModel` comes from 04 -- the same synthetic stack its own checks
        use, so the pruner here is exercised exactly as it was there,
        including the attention shape it needs for the KV placeholder.
        """

        def __init__(self):
            self.model = _FakeModel()
            self.calls = 0

        def reset(self):
            self.calls = 0

        def step(self, image, instruction):
            self.calls += 1
            self.model(torch.randn(1, 4, 16))       # a real forward to hook
            return np.zeros((2, 7), dtype=np.float32)


    _CLOSED = {"n": 0}


    class _StubEnv(GymnasiumEnv):
        def close(self):
            _CLOSED["n"] += 1


    def _stub_factory(task, ep_id):
        # Odd episodes succeed, so success rates are neither 0 nor 100.
        adapter = EnvAdapter(_StubEnv(solve_at=6 if ep_id % 2 else 10_000),
                             get_image=lambda obs: obs["cam"], max_steps=20)
        return adapter, "pick up the cup"


    policy = _StubVLA()
    print("check_policy:")
    check_policy(policy, image_shape=(64, 64, 3))

    tmp = tempfile.mkdtemp()
    try:
        out = run_campaign(policy, _stub_factory,
                           {"widowx_stack_cube": list(range(6))}, tmp,
                           model_name="stub")

        names = sorted(os.listdir(tmp))
        assert names == sorted(n for n, _ in CONDITIONS), names
        print("\\n  wrote:", ", ".join(names))

        # The depth conditions must have restored the stack afterwards.
        assert not hasattr(policy, "_depth")
        assert all(not isinstance(l, BypassDecoderLayer) for l in policy.model.layers)
        print("  decoder stack restored after the depth conditions")

        # And the files must pair, which is the only thing the analysis needs.
        def load(cond):
            p = os.path.join(tmp, cond, "widowx_stack_cube",
                             "results_widowx_stack_cube.json")
            with open(p) as fh:
                return {int(e["ep_id"]): bool(e["success"])
                        for e in json.load(fh)["episodes"]}

        base, pruned = load("baseline"), load("depth_prune4")
        assert set(base) == set(pruned) == set(range(6))
        print(f"  baseline and depth_prune4 pair on all {len(base)} episodes")

        # The recorded metadata has to say which condition it was.
        with open(os.path.join(tmp, "action_repeat2", "widowx_stack_cube",
                               "results_widowx_stack_cube.json")) as fh:
            meta = json.load(fh)
        assert meta["action_repeat"] == 2 and meta["condition"] == "action_repeat2"
        print("  condition metadata is in the file")

        # Every env the factory built must have been closed. SimplerEnv builds
        # one per episode; leaking them means a run that dies partway through
        # a condition with nothing in the traceback about renderers.
        expected = len(CONDITIONS) * 6
        assert _CLOSED["n"] == expected, (_CLOSED["n"], expected)
        print(f"  all {expected} envs closed ({len(CONDITIONS)} conditions x 6 episodes)")
    finally:
        shutil.rmtree(tmp)

    print("\\nthe driver works end to end; only the env builders need a simulator")
    '''),
    md("""
    ## Before trusting the first real cell

    In order, because each one makes the next meaningful.

    1. **Run the dry run above.** It costs seconds and covers everything except
       the two environment builders.
    2. **Run `check_policy`.** Note the native chunk length — a policy already
       emitting a chunk of 10 is amortised 10× before action repeat touches it,
       so its `action_repeat2` row is not comparable to a chunk-1 model's
       without saying so.
    3. **Run `baseline` alone and compare it to the number that model's own
       paper reports for that benchmark.** Every other row in the column is a
       difference against this one, so an error here is inherited by all seven.

       Not a formality: our OpenVLA baseline on `libero_spatial` came out at
       74.0% against a published 84.7%, a reproducible 10.7-point gap we never
       explained. Differences measured with the gap held fixed are still
       usable; absolute numbers are not comparable to the literature until it
       is understood. Better to find it before seven more runs than after.
    4. **Check that depth pruning actually got faster.** Bypassing N of L
       layers should cut `model_ms_per_infer` by roughly N/L. If the success
       rate moved and the latency did not, the layers were not really bypassed
       — check `find_decoder_layers`, do not conclude the method failed.
    5. **Then the remaining conditions.**

    One habit worth keeping: run `baseline` twice, on the same episode ids, at
    different times. Any episode that disagrees between the two is
    non-determinism, and it caps how small a difference the rest of the column
    can claim.

    ## Two things this notebook does not do

    Both are stated rather than worked around, because a silent workaround
    would be worse than the gap.

    **One camera view.** `run_episode` passes a single image to
    `policy.step(image, instruction)`. One of our own backbones (UniVLA) also
    consumes a wrist view, and its wrapper had to be given one. A model that
    needs a second view needs that plumbed through `EnvAdapter.get_image`, the
    hook, and the policy call — foveation's hook already handles a dict or a
    list of views, so the loop is the only piece to extend.

    **Bridge episodes are seeded here; ours were not.** Our own Bridge runs
    passed only `episode_id` to `reset`, letting the rest of the randomness
    fall where it fell; `build_bridge_env` above also sets `seed=ep_id`, the
    way the Fractal protocol always did. That makes conditions more exactly
    matched, not less, so it is the better default — but it means a Bridge
    column produced here is not bit-identical to ours. It does not need to be:
    every column is compared against its own baseline.
    """),
]
write("05_run_campaign.ipynb", cells05)

print("\ndone")

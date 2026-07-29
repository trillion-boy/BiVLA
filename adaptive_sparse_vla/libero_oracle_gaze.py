"""Ground-truth fovea placement for LIBERO, as a diagnostic upper bound.

Why this exists
---------------
The same foveation that *gained* OpenVLA +17.7/+18.8 points on SimplerEnv
Bridge costs it −16/−74 on `libero_spatial`. The leading explanation is fovea
*placement*: log-polar keeps the image centre sharp, Bridge's targets sit near
the centre, and `libero_spatial`'s do not — and within LIBERO the OpenVLA
losses track target eccentricity exactly (centre tasks hold at 5/5, right-
periphery tasks fall to 0/5).

This class tests that explanation directly by putting the fovea on the target
the simulator says is there. It reads privileged state no deployed policy has,
so it is **not** a method — it is the ceiling that a real, instruction-
conditioned gaze (the model's own attention over visual tokens) would be
chasing. If foveation-with-perfect-gaze still fails, the placement hypothesis
is dead and no amount of gaze engineering rescues it.

`MotionGaze` cannot answer this: frame differencing tracks whatever is moving,
which at the start of an episode is the arm and not the target, and it never
reads the instruction — so a null result there is ambiguous between "placement
does not matter" and "differencing cannot find the target".

Which object
------------
LIBERO's BDDL declares `(:obj_of_interest <moved> <destination>)` — for
`libero_spatial`, `(:obj_of_interest akita_black_bowl_1 plate_1)` against the
goal `(On akita_black_bowl_1 plate_1)`. Tracking the *moved* object covers the
whole episode by itself: before the grasp it is the thing to reach for, and
after the grasp it travels with the gripper toward the destination, so the
fovea follows the manipulation without needing a phase switch.

Verified against all 10 `libero_spatial` BDDLs: the target is `_1` in every
task and the distractor is `_2`, and each `_1`'s init region matches its
instruction exactly ("next to the ramekin" -> `next_to_ramekin_region`, "on the
stove" -> `flat_stove_1_cook_region`, and so on). The drawer task is the one
that reads differently — `(In akita_black_bowl_1 wooden_cabinet_1_top_region)`
with `(Open ...)`, and its distractor `_2` sits on the cabinet *top* — so it is
also the case where picking the wrong object would be least visible. The oracle
resolves it correctly.

Coordinate conventions (the part that silently goes wrong)
----------------------------------------------------------
Three flips compose here, and two of them cancel:

1. `project_points_from_world_to_camera` applies robosuite's camera-axis
   correction (`diag(1, -1, -1)`), i.e. OpenCV convention, and returns
   `(row, col)` with row 0 at the **top** of an upright image.
2. `macros.IMAGE_CONVENTION` is `"opengl"`, so robosuite does **not** flip
   what `mjr_readPixels` produced: `obs["agentview_image"]` has row 0 at the
   **bottom**.
3. `eval_libero.get_libero_image` applies `[::-1, ::-1]` to match the
   checkpoints' training preprocessing — a vertical flip *and* a horizontal
   mirror.

(2) and (3)'s vertical flips cancel, leaving only the mirror, so a projected
`(row, col)` lands in the policy's image at `(row, W-1-col)`. Foveation takes
its centre as `(x, y) = (col, row)`, hence the returned `(W-1-col, row)`.
`test_libero_oracle_gaze.py` pins this down against a simulated render
pipeline.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


def policy_pixel_from_projected(row: int, col: int, width: int) -> Tuple[float, float]:
    """Projected (row, col) -> fovea centre (x, y) in the policy's image.

    Split out from the class so the convention can be tested without a
    simulator (see the module docstring for the derivation).
    """
    return (float(width - 1 - col), float(row))


class LiberoOracleGaze:
    """Fovea centre from the simulator's ground-truth target pose.

    Mirrors `MotionGaze`'s interface (`reset()` / `update(frame)`) so the eval
    loop can swap one for the other. `update` ignores the frame it is handed —
    the pose comes from the simulator, not the pixels — and returns `None` if
    anything about the scene cannot be resolved, which makes the caller fall
    back to the fixed image centre rather than crash a multi-hour run.
    """

    def __init__(self, env, resolution: int, camera_name: str = "agentview",
                 target_index: int = 0):
        self.env = env
        self.resolution = int(resolution)
        self.camera_name = camera_name
        self.target_index = int(target_index)
        self._transform: Optional[np.ndarray] = None
        self._body_id: Optional[int] = None
        self._target_name: Optional[str] = None
        self._warned = False

    def reset(self) -> None:
        # The camera is static across an episode, but a new episode may rebuild
        # the sim, so drop the cached transform and body id.
        self._transform = None
        self._body_id = None

    # -- internals ---------------------------------------------------------
    def _robosuite_env(self):
        # LIBERO wraps the robosuite env one level deep (OffScreenRenderEnv.env).
        return getattr(self.env, "env", self.env)

    def _resolve(self) -> bool:
        from robosuite.utils.camera_utils import get_camera_transform_matrix

        renv = self._robosuite_env()
        names = list(getattr(renv, "obj_of_interest", []) or [])
        if not names:
            raise RuntimeError("env exposes no obj_of_interest")
        if self.target_index >= len(names):
            raise RuntimeError(
                f"target_index {self.target_index} out of range for "
                f"obj_of_interest={names}"
            )
        self._target_name = names[self.target_index]
        obj = renv.objects_dict[self._target_name]
        self._body_id = renv.sim.model.body_name2id(obj.root_body)
        self._transform = get_camera_transform_matrix(
            sim=renv.sim,
            camera_name=self.camera_name,
            camera_height=self.resolution,
            camera_width=self.resolution,
        )
        return True

    # -- public ------------------------------------------------------------
    def target_name(self) -> Optional[str]:
        return self._target_name

    def update(self, frame: Optional[np.ndarray] = None) -> Optional[Tuple[float, float]]:
        from robosuite.utils.camera_utils import project_points_from_world_to_camera

        try:
            if self._transform is None:
                self._resolve()
            renv = self._robosuite_env()
            pos = np.asarray(renv.sim.data.body_xpos[self._body_id], dtype=np.float64)
            row, col = project_points_from_world_to_camera(
                pos, self._transform, self.resolution, self.resolution
            )
            return policy_pixel_from_projected(int(row), int(col), self.resolution)
        except Exception as exc:  # never take down a multi-hour eval
            if not self._warned:
                self._warned = True
                print(f"[oracle-gaze] disabled, falling back to the image centre "
                      f"({type(exc).__name__}: {exc})", flush=True)
            return None

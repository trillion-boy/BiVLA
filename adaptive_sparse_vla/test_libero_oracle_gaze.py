"""Pin down the oracle-gaze coordinate convention without a simulator.

The projection itself is robosuite's; what this file exists to check is the
flip chain around it, which is derived by reading source rather than measured
and is exactly the kind of thing that silently produces a plausible-looking
but wrong fovea centre.

Two independent checks:

  [A] Simulate the render pipeline end to end. Put a marker at a known place
      in an upright image, apply MuJoCo's OpenGL bottom-up ordering, then
      `eval_libero.get_libero_image`'s [::-1, ::-1], and confirm the formula
      finds the marker where it actually ended up.

  [B] Run the real robosuite projection against a hand-built pinhole camera
      whose answer can be computed independently, so the (row, col) ordering
      and the axis correction are confirmed rather than assumed.

    python test_libero_oracle_gaze.py    # -> ALL 9 ORACLE-GAZE CHECKS PASSED
"""
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from libero_oracle_gaze import policy_pixel_from_projected  # noqa: E402

ok = 0


def check(label, cond, extra=""):
    global ok
    assert cond, f"FAIL: {label} {extra}"
    ok += 1
    print(f"  ok  {label} {extra}")


# --------------------------------------------------------------------- [A]
print("[A] flip chain: projected (row, col) -> pixel the policy actually sees")

H = W = 64
MARK_ROW, MARK_COL = 12, 50          # where the target is in an UPRIGHT image

upright = np.zeros((H, W, 3), dtype=np.uint8)
upright[MARK_ROW, MARK_COL] = 255

# MuJoCo's mjr_readPixels fills bottom-up, and robosuite's IMAGE_CONVENTION is
# "opengl" (convention=1), so obs["agentview_image"] keeps that ordering.
as_observed = upright[::-1]
# eval_libero.get_libero_image: vertical flip AND horizontal mirror.
policy_image = as_observed[::-1, ::-1]

rows, cols = np.nonzero(policy_image[:, :, 0])
actual_row, actual_col = int(rows[0]), int(cols[0])

cx, cy = policy_pixel_from_projected(MARK_ROW, MARK_COL, W)
check("formula matches where the marker really landed",
      (int(cy), int(cx)) == (actual_row, actual_col),
      f"formula=({int(cy)}, {int(cx)}) actual=({actual_row}, {actual_col})")
check("vertical flips cancel (row is unchanged)", int(cy) == MARK_ROW,
      f"cy={int(cy)} row={MARK_ROW}")
check("horizontal mirror survives", int(cx) == W - 1 - MARK_COL,
      f"cx={int(cx)}")
check("centre is a fixed point of the mirror",
      policy_pixel_from_projected(31, 31, 64)[0] == 32.0,
      "(so an on-centre target is unaffected by a convention slip)")
check("foveation takes (x, y) not (row, col)",
      policy_pixel_from_projected(0, 0, W) == (float(W - 1), 0.0))

# --------------------------------------------------------------------- [B]
print("\n[B] robosuite projection against a hand-built pinhole camera")

try:
    from robosuite.utils.camera_utils import project_points_from_world_to_camera
except ImportError:
    print("  -- robosuite unavailable, skipping [B]")
else:
    class _FakeModel:
        camera_names = ("agentview",)

        def camera_name2id(self, name):
            return 0

        cam_fovy = np.array([45.0])

    # Camera at (0, 0, 2) looking straight down -Z, with MuJoCo's camera frame
    # convention (+x right, +y up, +z backwards out of the screen).
    class _FakeData:
        cam_xpos = np.array([[0.0, 0.0, 2.0]])
        cam_xmat = np.eye(3).reshape(1, 9)

    class _FakeSim:
        model = _FakeModel()
        data = _FakeData()

    from robosuite.utils.camera_utils import get_camera_transform_matrix

    res = 64
    T = get_camera_transform_matrix(_FakeSim(), "agentview", res, res)

    # A point directly under the camera must land dead centre.
    row, col = project_points_from_world_to_camera(
        np.array([0.0, 0.0, 0.0]), T, res, res)
    check("point on the optical axis projects to the image centre",
          (int(row), int(col)) == (res // 2, res // 2), f"({row}, {col})")

    # +x in world is +x in camera, which is +col.
    _, col_px = project_points_from_world_to_camera(
        np.array([0.3, 0.0, 0.0]), T, res, res)
    check("world +x increases the column index", int(col_px) > res // 2,
          f"col={col_px}")

    # +y in world is UP in the camera frame, so after the axis correction
    # (+y down in OpenCV) it must DECREASE the row index.
    row_py, _ = project_points_from_world_to_camera(
        np.array([0.0, 0.3, 0.0]), T, res, res)
    check("world +y decreases the row index (OpenCV +y is down)",
          int(row_py) < res // 2, f"row={row_py}")

    # Composed with the flip chain: a target to the world +x side ends up on
    # the LEFT of the policy's image, because of the surviving mirror.
    cx_px, _ = policy_pixel_from_projected(int(row), int(col_px), res)
    check("world +x lands left-of-centre in the policy image (mirror)",
          cx_px < res // 2, f"cx={cx_px}")

print(f"\nALL {ok} ORACLE-GAZE CHECKS PASSED")

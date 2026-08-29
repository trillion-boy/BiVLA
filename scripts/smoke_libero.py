#!/usr/bin/env python3
"""One reset/render/step smoke test for the configured LIBERO installation."""

from __future__ import annotations

import os

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv


def main() -> None:
    suite = benchmark.get_benchmark_dict()["libero_spatial"]()
    task = suite.get_task(0)
    bddl = os.path.join(
        get_libero_path("bddl_files"), task.problem_folder, task.bddl_file
    )
    env = OffScreenRenderEnv(
        bddl_file_name=bddl, camera_heights=64, camera_widths=64
    )
    try:
        env.seed(0)
        env.reset()
        observation = env.set_init_state(suite.get_task_init_states(0)[0])
        observation, reward, done, _ = env.step([0, 0, 0, 0, 0, 0, -1])
        print(
            {
                "task": task.language,
                "image_shape": observation["agentview_image"].shape,
                "reward": reward,
                "done": done,
            }
        )
    finally:
        env.close()


if __name__ == "__main__":
    main()


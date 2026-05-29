"""Sanity check: assemble a scene and open the MuJoCo viewer.

Usage:
    python scripts/smoke_mujoco.py
"""

from __future__ import annotations

from _env import setup_env

setup_env()


def main() -> int:
    # TODO: build a default SceneSpec, assemble MJCF via scene_builder,
    #       load MuJoCoScene, and launch mujoco.viewer.launch_passive.
    raise NotImplementedError


if __name__ == "__main__":
    raise SystemExit(main())

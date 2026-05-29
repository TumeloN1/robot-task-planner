"""Inverse kinematics for the Panda end-effector (`grip_site`).

Primary path uses `mink` (QP-based, respects joint limits and a posture task).
A dependency-free damped-least-squares fallback is provided for environments
where `mink` is unavailable; the top-down grasp constraint keeps the fallback
well-behaved for the MVP.
"""

from __future__ import annotations

import numpy as np

# Fixed downward orientation (wxyz) for top-down grasps in the MVP.
TOP_DOWN_QUAT = np.array([0.0, 1.0, 0.0, 0.0])


def solve_ik(scene, target_pos: np.ndarray, target_quat: np.ndarray = TOP_DOWN_QUAT,
             *, use_mink: bool = True) -> np.ndarray | None:
    """Solve for arm joint angles placing `grip_site` at the target pose.

    Returns the 7 arm joint targets, or None if no solution is found.

    TODO:
      - mink path: build a FrameTask on grip_site + posture task, iterate.
      - fallback: damped least squares with mujoco.mj_jacSite.
    """
    raise NotImplementedError


def _dls_fallback(scene, target_pos, target_quat, *, damping=1e-2, iters=100):
    """Damped-least-squares Jacobian IK (no external dependencies)."""
    raise NotImplementedError

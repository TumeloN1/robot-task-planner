"""IK reachability + collision pre-check used by the validator.

Runs IK on a cloned `mjData` so the live simulation is never disturbed. This is
the precise layer of validation: it catches structurally valid but physically
impossible commands (target inside the table, unreachable pose, would-collide)
before the real arm moves, protecting both the sim state and the episode log.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class FeasibilityResult:
    reachable: bool
    collision_free: bool
    reason: str = ""

    @property
    def ok(self) -> bool:
        return self.reachable and self.collision_free


def check_pose_feasible(
    scene, target_pos: np.ndarray, target_quat: np.ndarray
) -> FeasibilityResult:
    """Return whether `grip_site` can reach `target_pos`/`target_quat` collision-free.

    TODO:
      - clone mjData, solve IK on the clone,
      - forward-kinematics the solution, verify site error < tol (reachable),
      - run collision detection (mj_forward + contact count) for collision_free.
    """
    raise NotImplementedError

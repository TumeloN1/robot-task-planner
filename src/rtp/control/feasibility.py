"""IK reachability + collision pre-check used by the validator.

Runs IK on a cloned `mjData` so the live simulation is never disturbed. This is
the precise layer of validation: it catches structurally valid but physically
impossible commands (target inside the table, unreachable pose, would-collide)
before the real arm moves, protecting both the sim state and the episode log.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import mujoco
import numpy as np

from rtp.control.ik import ARM_JOINTS, TOP_DOWN_QUAT, solve_ik

_ARM_BODIES = (
    "link0", "link1", "link2", "link3", "link4", "link5", "link6", "link7",
    "hand", "left_finger", "right_finger",
)
_PENETRATION_TOL = 0.005  # meters; deeper contacts count as a collision


@dataclass
class FeasibilityResult:
    reachable: bool
    collision_free: bool
    reason: str = ""

    @property
    def ok(self) -> bool:
        return self.reachable and self.collision_free


def _arm_geom_ids(model) -> set[int]:
    geoms: set[int] = set()
    for name in _ARM_BODIES:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if bid >= 0:
            start = model.body_geomadr[bid]
            geoms.update(range(start, start + model.body_geomnum[bid]))
    return geoms


def check_pose_feasible(
    scene, target_pos: np.ndarray, target_quat: np.ndarray = TOP_DOWN_QUAT
) -> FeasibilityResult:
    """Whether `grip_site` can reach the pose collision-free (checked off-line)."""
    q = solve_ik(scene, np.asarray(target_pos, dtype=float), target_quat)
    if q is None:
        return FeasibilityResult(False, False, "unreachable: no IK solution")

    model = scene.model
    d = copy.deepcopy(scene.data)
    qadr = [
        model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j)]
        for j in ARM_JOINTS
    ]
    d.qpos[qadr] = q
    mujoco.mj_forward(model, d)

    arm_geoms = _arm_geom_ids(model)
    worst = 0.0
    for c in d.contact[: d.ncon]:
        g1, g2 = int(c.geom1), int(c.geom2)
        involves_arm = (g1 in arm_geoms) ^ (g2 in arm_geoms)  # arm vs environment
        if involves_arm and c.dist < -_PENETRATION_TOL:
            worst = min(worst, c.dist)
    if worst < -_PENETRATION_TOL:
        return FeasibilityResult(True, False, f"collision: penetration {worst:.3f} m")
    return FeasibilityResult(True, True, "ok")

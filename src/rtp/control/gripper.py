"""Panda gripper control + grasp detection.

The menagerie Panda drives both fingers through tendon actuator `actuator8`
with ctrlrange [0, 255] (255 = open, 0 = closed). Grasp detection checks that
both fingertips are in contact with the target object; stability over time is
enforced separately by the temporal checkers in primitives/checks.py.
"""

from __future__ import annotations

import mujoco
import numpy as np

GRIPPER_OPEN = 255.0
GRIPPER_CLOSED = 0.0
_GRIPPER_ACTUATOR = "actuator8"
_FINGER_BODIES = ("left_finger", "right_finger")


def _body_geom_ids(model, body_name: str) -> set[int]:
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if bid < 0:
        return set()
    start = model.body_geomadr[bid]
    return set(range(start, start + model.body_geomnum[bid]))


class Gripper:
    def __init__(self, scene) -> None:
        self.scene = scene
        self.model = scene.model
        self.data = scene.data
        self._aid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, _GRIPPER_ACTUATOR)
        self._finger_geoms = {f: _body_geom_ids(self.model, f) for f in _FINGER_BODIES}

    def _set(self, value: float, settle: float) -> None:
        self.data.ctrl[self._aid] = value
        self.scene.step(max(1, int(settle / self.model.opt.timestep)))

    def open(self, *, settle: float = 0.5) -> None:
        self._set(GRIPPER_OPEN, settle)

    def close(self, *, settle: float = 0.6) -> None:
        self._set(GRIPPER_CLOSED, settle)

    def is_holding(self, object_id: str) -> bool:
        """True if both fingertips are in contact with the object's geoms."""
        obj_geoms = _body_geom_ids(self.model, object_id)
        if not obj_geoms:
            return False
        touching = {f: False for f in _FINGER_BODIES}
        for c in self.data.contact[: self.data.ncon]:
            g1, g2 = int(c.geom1), int(c.geom2)
            for finger, fgeoms in self._finger_geoms.items():
                if (g1 in fgeoms and g2 in obj_geoms) or (g2 in fgeoms and g1 in obj_geoms):
                    touching[finger] = True
        return all(touching.values())

    def finger_aperture(self) -> float:
        """Sum of the two finger joint positions (proxy for opening width)."""
        adr = [
            self.model.jnt_qposadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, j)]
            for j in ("finger_joint1", "finger_joint2")
        ]
        return float(np.sum(self.data.qpos[adr]))

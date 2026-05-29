"""Joint- and Cartesian-space motion for the Panda arm.

Writes targets to actuators actuator1..7 (position servos). Cartesian moves
resolve targets via `TopDownIK` and interpolate joint trajectories; every motion
ends in a settle phase so temporal postcondition checkers see a stable state.
The gripper actuator (actuator8) is never touched here.
"""

from __future__ import annotations

import mujoco
import numpy as np

from rtp.control.ik import ARM_JOINTS, TOP_DOWN_QUAT, TopDownIK

ARM_ACTUATORS = tuple(f"actuator{i}" for i in range(1, 8))


class ArmController:
    def __init__(self, scene, *, ik: TopDownIK | None = None) -> None:
        self.scene = scene
        self.model = scene.model
        self.data = scene.data
        self.ik = ik or TopDownIK(scene)
        self._aids = np.array([
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, a)
            for a in ARM_ACTUATORS
        ])
        self._qadr = np.array([
            self.model.jnt_qposadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, j)]
            for j in ARM_JOINTS
        ])

    def arm_qpos(self) -> np.ndarray:
        return self.data.qpos[self._qadr].copy()

    def move_to_joints(self, joint_targets: np.ndarray, *, duration: float = 2.0,
                       tol: float = 0.05) -> bool:
        """Interpolate arm actuator targets to `joint_targets`; return True if reached."""
        joint_targets = np.asarray(joint_targets, dtype=float)
        start = self.data.ctrl[self._aids].copy()
        n = max(1, int(duration / self.model.opt.timestep))
        for k in range(1, n + 1):
            alpha = k / n
            self.data.ctrl[self._aids] = (1 - alpha) * start + alpha * joint_targets
            self.scene.step(1)
        self.settle(0.3)
        return bool(np.max(np.abs(self.arm_qpos() - joint_targets)) < tol)

    def move_to_pose(self, target_pos: np.ndarray, target_quat: np.ndarray = TOP_DOWN_QUAT,
                     *, duration: float = 2.0) -> bool:
        """Solve IK then move; return False if no IK solution exists."""
        q = self.ik.solve(np.asarray(target_pos, dtype=float), target_quat)
        if q is None:
            return False
        return self.move_to_joints(q, duration=duration)

    def settle(self, seconds: float = 0.5) -> None:
        """Advance the sim (holding current ctrl) to let dynamics settle."""
        self.scene.step(max(1, int(seconds / self.model.opt.timestep)))

"""Joint- and Cartesian-space motion for the Panda arm.

Writes targets to actuators actuator1..7. Cartesian moves resolve targets via
`ik.solve_ik` and interpolate joint trajectories; every motion ends in a settle
phase so temporal postcondition checkers see a stable state.
"""

from __future__ import annotations

import numpy as np


class ArmController:
    def __init__(self, scene, control_hz: float = 50.0) -> None:
        self.scene = scene
        self.control_hz = control_hz

    def move_to_joints(self, joint_targets: np.ndarray, *, duration: float = 2.0) -> bool:
        """Interpolate to joint targets; return True when reached."""
        raise NotImplementedError

    def move_to_pose(self, target_pos: np.ndarray, target_quat: np.ndarray) -> bool:
        """Solve IK then move; return False if no IK solution."""
        raise NotImplementedError

    def settle(self, seconds: float = 0.5) -> None:
        """Advance the sim to let dynamics settle before checking postconditions."""
        raise NotImplementedError

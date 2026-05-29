"""Panda gripper control + grasp detection.

The menagerie Panda drives both fingers through tendon actuator `actuator8`
with ctrlrange [0, 255] (255 = open, 0 = closed). Grasp detection uses fingertip
contact plus an object-following test, and is evaluated temporally (see
primitives/checks.py) rather than on a single frame.
"""

from __future__ import annotations

GRIPPER_OPEN = 255.0
GRIPPER_CLOSED = 0.0


class Gripper:
    def __init__(self, scene, actuator_name: str = "actuator8") -> None:
        self.scene = scene
        self.actuator_name = actuator_name

    def open(self) -> None:
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError

    def is_holding(self, object_id: str) -> bool:
        """True if both fingertips contact `object_id` and it follows the hand."""
        raise NotImplementedError

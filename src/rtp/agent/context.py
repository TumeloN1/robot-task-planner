"""Execution context shared by every primitive.

Bundles the simulator, controllers, and perception, plus the mutable world
knowledge (latest `SceneState`, currently held object). Primitives receive this
as their first argument and never reach into globals.
"""

from __future__ import annotations

import numpy as np

from rtp.perception.api import ObjectState, SceneState

# Tabletop geometry / motion heights (world frame, meters).
TABLE_TOP_Z = 0.40
PREGRASP_OFFSET = 0.12
GRASP_OFFSET = 0.02
CARRY_Z = 0.66
PLACE_APPROACH_Z = 0.65
PLACE_Z = 0.52


class ExecutionContext:
    def __init__(self, scene, arm, gripper, perception, *, injector=None) -> None:
        self.scene = scene
        self.arm = arm
        self.gripper = gripper
        self.perception = perception
        self.injector = injector
        self.scene_state: SceneState | None = None
        self.held_object: str | None = None
        self.step = 0

    def perceive(self) -> SceneState:
        self.scene_state = self.perception.observe(self.step)
        self.step += 1
        return self.scene_state

    def resolve(self, query: str) -> ObjectState | None:
        """Resolve a name/id/description against current knowledge."""
        if self.scene_state is None:
            self.perceive()
        known = self.scene_state.all_known()
        q = query.strip().lower().replace("_", " ")
        for o in known:  # exact name or id
            if o.name.lower() == q or o.id.lower().replace("_", " ") == q:
                return o
        for o in known:  # all descriptor tokens present in the name or id
            hay = f"{o.name.lower()} {o.id.lower().replace('_', ' ')}"
            if all(tok in hay for tok in q.split()):
                return o
        return None

    def object_pos(self, object_id: str) -> np.ndarray:
        """Live ground-truth body position (used for motion targets)."""
        return self.scene.body_pos(object_id)

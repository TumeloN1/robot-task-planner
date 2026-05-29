"""Perception contract shared by ground-truth and (future) vision providers.

Partial observability is enforced from day one: a `SceneState` exposes only the
objects currently `visible`, plus a memory of previously seen ones. This keeps
the planner from assuming global omniscience, so swapping ground truth for a real
vision model later does not break the planner.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np


@dataclass
class Pose:
    """Rigid pose in the world frame."""

    position: np.ndarray  # (3,) xyz, meters
    quaternion: np.ndarray  # (4,) wxyz


@dataclass
class ObjectState:
    """A single perceived object.

    `visible` and `confidence` are first-class so that downstream prompts and
    primitives must reason about uncertainty. An occluded object has
    `visible=False`; `last_seen_step` lets the planner fall back to memory and
    decide whether to search (`find_object` / `look_around`).
    """

    id: str
    name: str  # human-facing label, e.g. "red mug"
    category: str  # e.g. "mug", "block", "tray", "table"
    color: str | None
    pose: Pose
    aabb: np.ndarray  # (2, 3) axis-aligned bbox: [min_xyz, max_xyz]
    visible: bool = True
    confidence: float = 1.0
    last_seen_step: int | None = None
    attributes: dict = field(default_factory=dict)

    @property
    def graspable(self) -> bool:
        """Whether a primitive is allowed to attempt a grasp on this object."""
        return self.category not in {"table", "tray"}


@dataclass
class SceneState:
    """Snapshot of what the agent currently knows about the world.

    `visible_objects` are observed this step. `remembered_objects` carries the
    last-seen state of objects not currently visible (e.g. now occluded).
    """

    step: int
    visible_objects: list[ObjectState]
    remembered_objects: list[ObjectState] = field(default_factory=list)
    proprio: np.ndarray | None = None  # robot joint positions/velocities
    rgb: np.ndarray | None = None  # optional (H, W, 3) uint8
    depth: np.ndarray | None = None  # optional (H, W) float32 meters

    def all_known(self) -> list[ObjectState]:
        """Visible objects plus remembered ones (visible take precedence)."""
        seen = {o.id for o in self.visible_objects}
        return self.visible_objects + [o for o in self.remembered_objects if o.id not in seen]

    def find(self, name_or_id: str) -> ObjectState | None:
        for o in self.all_known():
            if o.id == name_or_id or o.name == name_or_id:
                return o
        return None


class PerceptionProvider(Protocol):
    """Interface every perception backend implements (GT now, vision later)."""

    def observe(self, step: int) -> SceneState:
        """Return the current `SceneState` for the given step index."""
        ...

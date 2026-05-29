"""Scene specification dataclasses.

A `SceneSpec` is the serializable description (JSON) produced by the procedural
generator and consumed by `scene_builder` to assemble the MJCF model.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ObjectSpec:
    """One manipulable/static object placed on the table."""

    id: str
    category: str  # "mug" | "block" | "tray"
    color: str
    position: tuple[float, float, float]
    yaw: float = 0.0  # rotation about z (radians)

    @property
    def name(self) -> str:
        return f"{self.color} {self.category}"


@dataclass
class SceneSpec:
    """A full procedurally generated scene."""

    seed: int
    objects: list[ObjectSpec] = field(default_factory=list)

    def to_dict(self) -> dict:
        raise NotImplementedError

    @classmethod
    def from_dict(cls, data: dict) -> "SceneSpec":
        raise NotImplementedError

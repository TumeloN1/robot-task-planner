"""Scene specification dataclasses.

A `SceneSpec` is the serializable description (JSON) produced by the procedural
generator and consumed by `scene_builder` to assemble the MJCF model.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

# Categories that are spawned as static (no free joint) placement targets.
STATIC_CATEGORIES = {"tray"}


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

    @property
    def is_static(self) -> bool:
        return self.category in STATIC_CATEGORIES

    def to_dict(self) -> dict:
        d = asdict(self)
        d["position"] = list(self.position)
        return d

    @classmethod
    def from_dict(cls, data: dict) -> ObjectSpec:
        return cls(
            id=data["id"],
            category=data["category"],
            color=data["color"],
            position=tuple(data["position"]),
            yaw=data.get("yaw", 0.0),
        )


@dataclass
class SceneSpec:
    """A full procedurally generated scene."""

    seed: int
    objects: list[ObjectSpec] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"seed": self.seed, "objects": [o.to_dict() for o in self.objects]}

    @classmethod
    def from_dict(cls, data: dict) -> SceneSpec:
        return cls(
            seed=data["seed"],
            objects=[ObjectSpec.from_dict(o) for o in data.get("objects", [])],
        )

    def by_id(self, object_id: str) -> ObjectSpec | None:
        for o in self.objects:
            if o.id == object_id:
                return o
        return None

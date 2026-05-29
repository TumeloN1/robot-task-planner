"""Procedural scene + instruction generation.

Produces collision-free `SceneSpec`s and pairs instruction templates
(configs/tasks.yaml) with their programmatic success predicates. Seeded for
reproducibility.
"""

from __future__ import annotations

from dataclasses import dataclass

from rtp.sim.objects import SceneSpec


@dataclass
class Task:
    instruction: str
    predicate: dict  # {relation, subject, target, ...} resolved against the scene
    scene: SceneSpec


def generate_scene(seed: int, *, num_objects_range=(2, 5), collision_margin_m=0.03) -> SceneSpec:
    """Sample a collision-free tabletop scene (rejection sampling on AABBs)."""
    raise NotImplementedError


def generate_task(seed: int, *, templates: list[dict], vocabulary: dict) -> Task:
    """Sample a scene + an instruction template + its resolved success predicate."""
    raise NotImplementedError

"""Procedural scene + instruction generation.

Produces collision-free `SceneSpec`s and pairs instruction templates
(configs/tasks.yaml) with their programmatic success predicates. Seeded for
reproducibility.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rtp.sim.objects import ObjectSpec, SceneSpec

# Table-top sampling region (within the workspace + table footprint).
_X_RANGE = (0.38, 0.66)
_Y_RANGE = (-0.28, 0.28)
_MOVABLE_Z = 0.43
_TRAY_Z = 0.41


@dataclass
class Task:
    instruction: str
    predicate: dict  # {relation, subject, target, ...}
    scene: SceneSpec


def _sample_positions(rng: np.random.Generator, n: int, *, margin: float) -> list[tuple]:
    positions: list[np.ndarray] = []
    attempts = 0
    while len(positions) < n and attempts < 1000:
        attempts += 1
        p = np.array([rng.uniform(*_X_RANGE), rng.uniform(*_Y_RANGE)])
        if all(np.linalg.norm(p - q) > margin for q in positions):
            positions.append(p)
    if len(positions) < n:
        raise RuntimeError("could not place objects without collisions; loosen the margin")
    return [tuple(p) for p in positions]


def generate_scene(spec_objects: list[tuple[str, str, str]], *, seed: int,
                   collision_margin_m: float = 0.14) -> SceneSpec:
    """Build a collision-free scene from (id, category, color) tuples."""
    rng = np.random.default_rng(seed)
    xy = _sample_positions(rng, len(spec_objects), margin=collision_margin_m)
    objects = []
    for (oid, category, color), (x, y) in zip(spec_objects, xy, strict=True):
        z = _TRAY_Z if category == "tray" else _MOVABLE_Z
        objects.append(ObjectSpec(id=oid, category=category, color=color, position=(x, y, z)))
    return SceneSpec(seed=seed, objects=objects)


def generate_task(seed: int, *, templates: list[dict], vocabulary: dict) -> Task:
    """Sample a scene + instruction template + resolved success predicate."""
    rng = np.random.default_rng(seed)
    template = templates[rng.integers(len(templates))]
    colors = vocabulary["colors"]
    movables = vocabulary["objects"]

    if template["id"] == "stack":
        ca, cb = rng.choice(colors, size=2, replace=False)
        a_cat, b_cat = rng.choice(movables), rng.choice(movables)
        spec = [(f"{ca}_{a_cat}", a_cat, ca), (f"{cb}_{b_cat}", b_cat, cb)]
        scene = generate_scene(spec, seed=seed)
        instruction = template["instruction"].format(a=f"{ca} {a_cat}", b=f"{cb} {b_cat}")
        predicate = {"relation": "on", "subject": f"{ca} {a_cat}", "target": f"{cb} {b_cat}"}
        return Task(instruction, predicate, scene)

    # Templates with a {color}/{object} subject and a tray target.
    color = rng.choice(colors)
    cat = rng.choice(movables)
    subject = f"{color} {cat}"
    spec = [(f"{color}_{cat}", cat, color), ("tray", "tray", "blue")]
    # Optional distractor of a different color.
    other = rng.choice([c for c in colors if c != color])
    spec.append((f"{other}_{cat}", cat, other))
    scene = generate_scene(spec, seed=seed)

    instruction = template["instruction"].format(color=color, object=cat, target="tray")
    pred = dict(template["predicate"])
    pred["subject"] = subject
    pred["target"] = "tray"
    return Task(instruction, pred, scene)

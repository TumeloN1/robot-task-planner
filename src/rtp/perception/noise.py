"""Perception augmentation: wraps a provider to inject perception failures.

Used by the failure-injection probe (configs/run.yaml). Injectors:
  - label swap: swap ids/names of two similar-colored objects (wrong-object).
  - pose jitter: add Gaussian noise to positions.
  - drop visibility: force an object visible=False to trigger search behavior.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from rtp.perception.api import PerceptionProvider, SceneState


class NoisyPerception:
    """Decorates a `PerceptionProvider` with configurable perceptual errors."""

    def __init__(self, inner: PerceptionProvider, *, rng: np.random.Generator | None = None,
                 swap_prob: float = 0.0, pos_jitter_m: float = 0.0,
                 drop_visibility_prob: float = 0.0) -> None:
        self.inner = inner
        self.rng = rng or np.random.default_rng()
        self.swap_prob = swap_prob
        self.pos_jitter_m = pos_jitter_m
        self.drop_visibility_prob = drop_visibility_prob

    def observe(self, step: int) -> SceneState:
        scene = self.inner.observe(step)
        objs = [dataclasses.replace(o) for o in scene.visible_objects]

        if self.pos_jitter_m > 0:
            for o in objs:
                noise = self.rng.normal(0, self.pos_jitter_m, size=3)
                o.pose = dataclasses.replace(o.pose, position=o.pose.position + noise)

        if self.drop_visibility_prob > 0:
            objs = [o for o in objs if self.rng.random() >= self.drop_visibility_prob]

        if self.swap_prob > 0 and len(objs) >= 2 and self.rng.random() < self.swap_prob:
            i, j = self.rng.choice(len(objs), size=2, replace=False)
            objs[i].id, objs[j].id = objs[j].id, objs[i].id
            objs[i].name, objs[j].name = objs[j].name, objs[i].name

        return SceneState(step=step, visible_objects=objs,
                          remembered_objects=scene.remembered_objects)

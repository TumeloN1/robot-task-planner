"""Perception augmentation: wraps a provider to inject perception failures.

Used by the failure-injection probe (configs/run.yaml). Injectors:
  - label swap: swap ids of two similar-colored objects (wrong-object failures).
  - pose jitter: add Gaussian noise to positions.
  - drop visibility: force an object visible=False to trigger search behavior.
"""

from __future__ import annotations

from rtp.perception.api import PerceptionProvider, SceneState


class NoisyPerception:
    """Decorates a `PerceptionProvider` with configurable perceptual errors."""

    def __init__(self, inner: PerceptionProvider, *, rng, swap_prob: float = 0.0,
                 pos_jitter_m: float = 0.0, drop_visibility_prob: float = 0.0) -> None:
        self.inner = inner
        self.rng = rng
        self.swap_prob = swap_prob
        self.pos_jitter_m = pos_jitter_m
        self.drop_visibility_prob = drop_visibility_prob

    def observe(self, step: int) -> SceneState:
        """Observe via inner provider, then apply the configured perturbations."""
        raise NotImplementedError

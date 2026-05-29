"""Failure injection controllers - the self-correction probe.

Each injector deterministically (given a seed) produces a failure case the
diagnosis/replan loop must handle, enabling per-failure-type recovery metrics.
Rates are configured in configs/run.yaml.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class InjectionPlan:
    """Which failures to inject during an episode (sampled per scene)."""

    grasp_slip: bool = False
    perception_confusion: bool = False
    target_blocked: bool = False
    actuation_noise: bool = False


class FailureInjector:
    """Applies injected failures at the right hooks during an episode."""

    def __init__(self, plan: InjectionPlan, *, rng) -> None:
        self.plan = plan
        self.rng = rng

    def maybe_slip(self, scene, held_object_id: str) -> bool:
        """If grasp_slip is active, force-release mid-transport. Returns whether slipped."""
        raise NotImplementedError

    def perturb_controls(self, ctrl):
        """If actuation_noise is active, add noise to controller targets."""
        raise NotImplementedError

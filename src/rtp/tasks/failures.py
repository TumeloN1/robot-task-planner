"""Failure injection controllers - the self-correction probe.

Each injector deterministically (given a seed) produces a failure case the
diagnosis/replan loop must handle, enabling per-failure-type recovery metrics.
Rates are configured in configs/run.yaml.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class InjectionPlan:
    """Which failures to inject during an episode (sampled per scene)."""

    grasp_slip: bool = False
    perception_confusion: bool = False
    target_blocked: bool = False
    actuation_noise: bool = False

    def active(self) -> list[str]:
        return [k for k, v in self.__dict__.items() if v]

    @classmethod
    def sample(cls, rng: np.random.Generator, rates: dict) -> InjectionPlan:
        return cls(
            grasp_slip=rng.random() < rates.get("grasp_slip", 0.0),
            perception_confusion=rng.random() < rates.get("perception_confusion", 0.0),
            target_blocked=rng.random() < rates.get("target_blocked", 0.0),
            actuation_noise=rng.random() < rates.get("actuation_noise", 0.0),
        )


class FailureInjector:
    """Applies injected failures at the right hooks during an episode."""

    def __init__(self, plan: InjectionPlan, *, rng: np.random.Generator) -> None:
        self.plan = plan
        self.rng = rng
        self._slipped_once = False

    def maybe_slip(self, scene, held_object_id: str) -> bool:
        """If grasp_slip is active, force-release the object once mid-transport."""
        if self.plan.grasp_slip and not self._slipped_once:
            self._slipped_once = True
            return True
        return False

    def perturb_controls(self, ctrl: np.ndarray) -> np.ndarray:
        """If actuation_noise is active, add noise to controller targets."""
        if not self.plan.actuation_noise:
            return ctrl
        return ctrl + self.rng.normal(0, 0.02, size=ctrl.shape)

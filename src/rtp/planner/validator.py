"""Deterministic plan validation, cheapest checks first.

1. Schema/grounding : tool exists, args match the model, object refs resolve,
                      named targets are derivable.
2. Physics sanity   : targets within workspace + above table; graspable category;
                      AABB checks (fast reject).
3. IK feasibility   : reachable + collision-free on a cloned mjData (precise).
4. Loop detection   : reject a replan near-identical to one that already failed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from rtp.perception.api import SceneState
from rtp.planner.schema import Plan


@dataclass
class ValidationResult:
    valid: bool
    errors: list[str] = field(default_factory=list)

    def as_repair_text(self) -> str:
        """Concise, directive error text to feed back to the model for repair."""
        return "; ".join(self.errors)


def validate_plan(plan: Plan, scene: SceneState, *, scene_ctx=None,
                  failed_plans: list[Plan] | None = None) -> ValidationResult:
    """Run the four validation layers and aggregate errors.

    TODO:
      - schema/grounding against PRIMITIVES + scene.find,
      - physics sanity against configs/sim.yaml workspace,
      - IK feasibility via control.feasibility.check_pose_feasible,
      - loop detection against failed_plans.
    """
    raise NotImplementedError

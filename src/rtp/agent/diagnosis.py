"""Failure classification and concise replan-context construction.

Emits a ruthlessly compact `FailureContext` (never raw state dumps): what was
attempted, the one-line observed result, and a directive `must_change`. It also
carries a bounded count of prior attempts to avoid context bloat that causes the
model to loop.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class FailureType(str, Enum):
    GRASP_FAILED = "grasp_failed"
    OBJECT_SLIPPED = "object_slipped"
    TARGET_BLOCKED = "target_blocked"
    WRONG_OBJECT = "wrong_object"
    UNREACHABLE = "unreachable"
    NOT_VISIBLE = "not_visible"
    GOAL_NOT_MET = "goal_not_met"
    TIMEOUT = "timeout"


_DIRECTIVES = {
    FailureType.GRASP_FAILED: "The grasp did not secure the object. Re-approach and grasp "
                              "again, or target a different graspable object; do not repeat "
                              "the identical sequence.",
    FailureType.OBJECT_SLIPPED: "The object slipped during transport. Re-grasp it before "
                                "moving, and verify the grasp with check_grasp_success.",
    FailureType.UNREACHABLE: "The pose was unreachable. Re-find the object or choose a "
                             "different target.",
    FailureType.NOT_VISIBLE: "The object was not visible. Call look_around or find_object "
                             "before acting on it.",
    FailureType.TARGET_BLOCKED: "The target location is blocked. Clear it or choose another "
                                "placement.",
    FailureType.WRONG_OBJECT: "A different object than intended was handled. Re-find the "
                              "correct object by its description.",
    FailureType.GOAL_NOT_MET: "The goal relation was not satisfied. Reconsider placement and "
                              "redo the relevant steps.",
}


@dataclass
class FailureContext:
    failure_type: FailureType
    attempted: str
    result: str
    prior_attempts: int = 0

    @property
    def must_change(self) -> str:
        return _DIRECTIVES.get(self.failure_type, "Change your approach; do not repeat the "
                               "previous failed steps.")

    def render(self) -> str:
        return (
            f"- Attempted: {self.attempted}\n"
            f"- Result: {self.result}\n"
            f"- Prior failed attempts this episode: {self.prior_attempts}\n"
            f"- You MUST change: {self.must_change}"
        )


def classify(tool: str, result) -> FailureType:
    """Map a failed primitive/check result to a FailureType."""
    msg = (result.message or "").lower()
    if result.info.get("slipped"):
        return FailureType.OBJECT_SLIPPED
    if tool == "grasp":
        return FailureType.GRASP_FAILED
    if "not visible" in msg:
        return FailureType.NOT_VISIBLE
    if "unreachable" in msg or "infeasible" in msg:
        return FailureType.UNREACHABLE
    if tool == "check_grasp_success":
        return FailureType.OBJECT_SLIPPED
    if tool == "check_task_success":
        return FailureType.GOAL_NOT_MET
    return FailureType.TIMEOUT

"""Failure classification and concise replan-context construction.

Emits a ruthlessly compact `FailureContext` (never raw state dumps): what was
attempted, the one-line observed result, and a directive `must_change`. It also
carries a bounded summary of prior attempts to avoid context bloat / attention
dilution that causes the model to loop.
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
    TIMEOUT = "timeout"


@dataclass
class FailureContext:
    failure_type: FailureType
    attempted: str  # e.g. "grasp(red mug) at top-down pose"
    result: str  # one line, e.g. "object not held after settle window"
    must_change: str  # directive, e.g. "change approach coords; do not repeat pose"
    prior_attempts: int = 0

    def render(self) -> str:
        """Compact text block for the replan prompt."""
        raise NotImplementedError


def classify(step_result, scene) -> FailureType:
    """Map a failed primitive/check result + scene to a FailureType."""
    raise NotImplementedError


def rule_based_recovery(failure: FailureContext):
    """Return a cheap deterministic recovery action, or None to defer to the LLM."""
    raise NotImplementedError

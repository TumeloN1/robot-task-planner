"""World / episode state tracking shared across the agent loop."""

from __future__ import annotations

from dataclasses import dataclass, field

from rtp.planner.schema import Plan


@dataclass
class StepRecord:
    """One executed step: the tool call, its result, and the check outcome."""

    tool: str
    args: dict
    success: bool
    postcondition_pass: bool | None
    failure_label: str | None
    info: dict = field(default_factory=dict)


@dataclass
class EpisodeState:
    instruction: str
    steps: list[StepRecord] = field(default_factory=list)
    failed_plans: list[Plan] = field(default_factory=list)
    num_replans: int = 0
    success: bool = False
    injected_failures: list[str] = field(default_factory=list)

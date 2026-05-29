"""Offline scripted planner (no LLM).

Useful for tests and deterministic demos: returns a fixed plan or delegates to a
callable. Implements the same `propose` interface as `GeminiPlanner`.
"""

from __future__ import annotations

from collections.abc import Callable

from rtp.perception.api import SceneState
from rtp.planner.schema import Plan


class ScriptedPlanner:
    def __init__(self, plan_or_fn: Plan | Callable[..., Plan]) -> None:
        self._src = plan_or_fn

    def propose(self, instruction: str, scene: SceneState,
                failure_context: str | None = None) -> Plan:
        if callable(self._src):
            return self._src(instruction, scene, failure_context)
        return self._src

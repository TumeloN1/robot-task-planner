"""Prompt construction for the planner.

The prompt contains the tool catalog (from the registry), the current
`SceneState` (only visible objects + last-seen memory), the instruction, and -
on a replan - a concise `FailureContext` (never raw logs).
"""

from __future__ import annotations

from rtp.perception.api import SceneState

SYSTEM_PROMPT = """You are a task planner for a Franka Panda robot arm on a tabletop.
You output a JSON plan: a short `rationale` then an ordered `tool_calls` list.
Use only the provided tools. You may not assume you can see objects that are not
listed as visible; use find_object or look_around to locate them first.
Grasps are top-down only. After acting, include the appropriate check_* tool.
"""


def build_tool_catalog() -> str:
    """Render the registered primitives + their argument schemas for the prompt."""
    raise NotImplementedError


def build_planning_prompt(instruction: str, scene: SceneState,
                          failure_context: str | None = None) -> str:
    """Assemble the full prompt string for an initial plan or a replan."""
    raise NotImplementedError

"""Prompt construction for the planner.

The prompt contains the tool catalog (from the registry), the current
`SceneState` (only visible objects + last-seen memory), the instruction, and -
on a replan - a concise failure context (never raw logs).
"""

from __future__ import annotations

import rtp.primitives  # noqa: F401  (populates the registry)
from rtp.perception.api import SceneState
from rtp.primitives.registry import PRIMITIVES

SYSTEM_PROMPT = """You are a task planner for a Franka Panda robot arm on a tabletop.
Output a JSON plan with a short `rationale` then an ordered `tool_calls` list.
Rules:
- Use ONLY the listed tools and argument names.
- You may only act on objects listed as visible. If a needed object is not
  visible, first call find_object or look_around.
- Grasps are top-down only. Grasp before moving; release after placing.
- After a grasp, call check_grasp_success. End with check_task_success.
- On a replan, your rationale MUST state what you are changing and why; do not
  repeat a previously failed action unchanged.
"""

OUTPUT_FORMAT = """OUTPUT FORMAT (return ONLY this JSON object, no prose, no markdown):
{
  "rationale": "<one or two sentences>",
  "tool_calls": [
    {"tool": "<tool name>", "args": {<argument name>: <value>, ...}},
    ...
  ]
}"""


def build_tool_catalog() -> str:
    lines = []
    for name, spec in PRIMITIVES.items():
        props = spec.args_model.model_json_schema().get("properties", {})
        args = ", ".join(props.keys()) if props else "(none)"
        lines.append(f"- {name}(args: {args}): {spec.description}")
    return "\n".join(lines)


def _describe_scene(scene: SceneState) -> str:
    if not scene.visible_objects:
        lines = ["(no objects currently visible)"]
    else:
        lines = []
        for o in scene.visible_objects:
            p = o.pose.position.round(3).tolist()
            lines.append(f"- {o.name} (id={o.id}, category={o.category}) at {p}")
    if scene.remembered_objects:
        seen = ", ".join(o.name for o in scene.remembered_objects)
        lines.append(f"(previously seen but not currently visible: {seen})")
    return "\n".join(lines)


def build_planning_prompt(instruction: str, scene: SceneState,
                          failure_context: str | None = None) -> str:
    parts = [
        SYSTEM_PROMPT,
        "AVAILABLE TOOLS:\n" + build_tool_catalog(),
        "VISIBLE SCENE:\n" + _describe_scene(scene),
        f"INSTRUCTION: {instruction}",
    ]
    if failure_context:
        parts.append("PREVIOUS ATTEMPT FAILED:\n" + failure_context)
    parts.append(OUTPUT_FORMAT)
    parts.append("Produce the JSON plan now.")
    return "\n\n".join(parts)

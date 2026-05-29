"""Sanity check: instruction -> Gemini plan -> validation (no execution).

Usage:
    python scripts/smoke_gemini.py "Put the red mug on the tray"
"""

from __future__ import annotations

import sys

from _env import planner_kwargs, setup_env

setup_env()

from rtp.agent.build import build_agent  # noqa: E402
from rtp.planner.gemini_client import GeminiPlanner  # noqa: E402
from rtp.planner.validator import validate_plan  # noqa: E402
from rtp.sim.scene_builder import default_scene_spec  # noqa: E402


def main(argv: list[str]) -> int:
    instruction = argv[1] if len(argv) > 1 else "Put the red mug on the tray"
    planner = GeminiPlanner(**planner_kwargs())
    ctx, _ = build_agent(default_scene_spec(), planner)
    scene_state = ctx.perceive()

    print(f"Instruction: {instruction}")
    print("Visible:", [o.name for o in scene_state.visible_objects])

    plan = planner.propose(instruction, scene_state)
    print("\nrationale:", plan.rationale)
    for c in plan.tool_calls:
        print(f"  {c.tool}({c.args})")

    vres = validate_plan(plan, scene_state, scene_ctx=ctx)
    print("\nvalid:", vres.valid)
    if not vres.valid:
        print("errors:", vres.as_repair_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

"""Run a single instruction end-to-end through the closed loop (live Gemini).

Usage:
    python scripts/run_episode.py "Put the red mug on the tray"
"""

from __future__ import annotations

import sys

from _env import planner_kwargs, setup_env

setup_env()

from rtp.agent.build import build_agent  # noqa: E402
from rtp.planner.gemini_client import GeminiPlanner  # noqa: E402
from rtp.sim.scene_builder import default_scene_spec  # noqa: E402


def main(argv: list[str]) -> int:
    instruction = argv[1] if len(argv) > 1 else "Put the red mug on the tray"
    planner = GeminiPlanner(**planner_kwargs())
    _, loop = build_agent(default_scene_spec(), planner)

    episode = loop.run(instruction)

    print(f"\nInstruction: {instruction}")
    print(f"SUCCESS: {episode.success}  replans: {episode.num_replans}  "
          f"steps: {len(episode.steps)}")
    for i, s in enumerate(episode.steps):
        flag = "ok" if s.success else f"FAIL[{s.failure_label}]"
        print(f"  {i:2d} {s.tool}({s.args}) -> {flag}")
    return 0 if episode.success else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

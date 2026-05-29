"""Run a single instruction end-to-end through the closed loop.

Usage:
    python scripts/run_episode.py "Put the red mug on the tray"
"""

from __future__ import annotations

import sys

from _env import setup_env

setup_env()


def main(argv: list[str]) -> int:
    instruction = argv[1] if len(argv) > 1 else "Put the red mug on the tray"
    # TODO: assemble scene, build perception/planner/validator/executor/recorder,
    #       construct AgentLoop, run(instruction), print the EpisodeState summary.
    print(f"Instruction: {instruction}")
    raise NotImplementedError


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

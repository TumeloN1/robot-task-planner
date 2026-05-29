"""Sanity check: instruction -> validated plan (no execution).

Usage:
    python scripts/smoke_gemini.py "Put the red mug on the tray"
"""

from __future__ import annotations

import sys

from _env import setup_env

setup_env()


def main(argv: list[str]) -> int:
    instruction = argv[1] if len(argv) > 1 else "Put the red mug on the tray"
    # TODO: build a mock SceneState, prompt, call GeminiPlanner.plan,
    #       run validate_plan, and print the plan + validation result.
    print(f"Instruction: {instruction}")
    raise NotImplementedError


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

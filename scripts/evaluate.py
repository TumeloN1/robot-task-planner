"""Batch-evaluate the agent over generated scenes and print metrics.

Usage:
    python scripts/evaluate.py
"""

from __future__ import annotations

from _env import setup_env

setup_env()


def main() -> int:
    # TODO: load scenes from data/scenes/, run AgentLoop on each (with failure
    #       injection per configs/run.yaml), aggregate via eval.metrics, print.
    raise NotImplementedError


if __name__ == "__main__":
    raise SystemExit(main())

"""Generate a dataset of scenes + tasks as JSON under data/scenes/.

Usage:
    python scripts/generate_scenes.py
"""

from __future__ import annotations

from _env import setup_env

setup_env()


def main() -> int:
    # TODO: load configs/run.yaml + configs/tasks.yaml, loop num_scenes,
    #       call tasks.generator.generate_task, serialize each to data/scenes/.
    raise NotImplementedError


if __name__ == "__main__":
    raise SystemExit(main())

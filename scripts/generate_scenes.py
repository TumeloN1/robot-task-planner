"""Generate a dataset of scenes + tasks as JSON under data/scenes/.

Usage:
    python scripts/generate_scenes.py [n]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

from _env import setup_env

setup_env()

from rtp.tasks.generator import generate_task  # noqa: E402

CONFIGS = Path(__file__).resolve().parent.parent / "configs"


def main(argv: list[str]) -> int:
    tasks_cfg = yaml.safe_load((CONFIGS / "tasks.yaml").read_text())
    run_cfg = yaml.safe_load((CONFIGS / "run.yaml").read_text())
    n = int(argv[1]) if len(argv) > 1 else run_cfg.get("num_scenes", 10)
    base_seed = run_cfg.get("seed", 0)

    out_dir = Path("data") / "scenes"
    out_dir.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        task = generate_task(base_seed + i, templates=tasks_cfg["templates"],
                             vocabulary=tasks_cfg["vocabulary"])
        record = {
            "instruction": task.instruction,
            "predicate": task.predicate,
            "scene": task.scene.to_dict(),
        }
        (out_dir / f"task_{i:04d}.json").write_text(json.dumps(record, indent=2))
    print(f"Wrote {n} tasks to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

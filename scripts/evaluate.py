"""Batch-evaluate the agent over generated scenes and print metrics.

Usage:
    python scripts/evaluate.py [n]   # n = number of tasks (default: all found)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import yaml

from _env import planner_kwargs, setup_env

setup_env()

from rtp.agent.build import build_agent  # noqa: E402
from rtp.data.recorder import EpisodeRecorder  # noqa: E402
from rtp.eval.metrics import compute_metrics  # noqa: E402
from rtp.planner.gemini_client import GeminiPlanner  # noqa: E402
from rtp.sim.objects import SceneSpec  # noqa: E402
from rtp.tasks import predicates  # noqa: E402
from rtp.tasks.failures import FailureInjector, InjectionPlan  # noqa: E402

CONFIGS = Path(__file__).resolve().parent.parent / "configs"


def _ground_truth_success(ctx, predicate: dict) -> bool:
    ctx.perceive()
    subj = ctx.resolve(predicate["subject"])
    tgt = ctx.resolve(predicate["target"])
    if subj is None or tgt is None:
        return False
    kwargs = {"threshold_m": predicate["threshold_m"]} if "threshold_m" in predicate else {}
    try:
        return predicates.evaluate(predicate["relation"], subj, tgt, **kwargs)
    except ValueError:
        return False


def main(argv: list[str]) -> int:
    run_cfg = yaml.safe_load((CONFIGS / "run.yaml").read_text())
    rates = run_cfg.get("failure_injection", {})
    scenes_dir = Path("data") / "scenes"
    task_files = sorted(scenes_dir.glob("task_*.json"))
    if len(argv) > 1:
        task_files = task_files[: int(argv[1])]
    if not task_files:
        print("No tasks found. Run scripts/generate_scenes.py first.")
        return 1

    planner = GeminiPlanner(**planner_kwargs())
    recorder = EpisodeRecorder(Path(run_cfg["logging"]["out_dir"]) / "episodes.hdf5")
    episodes = []

    for tf in task_files:
        rec = json.loads(tf.read_text())
        scene_spec = SceneSpec.from_dict(rec["scene"])
        rng = np.random.default_rng(scene_spec.seed)
        inj_plan = InjectionPlan.sample(rng, rates)
        injector = FailureInjector(inj_plan, rng=rng)

        ctx, loop = build_agent(scene_spec, planner, injector=injector, recorder=recorder)
        try:
            episode = loop.run(rec["instruction"], seed=scene_spec.seed)
            episode.injected_failures = inj_plan.active()
            episode.success = _ground_truth_success(ctx, rec["predicate"])
        except Exception as e:  # isolate transient API/runtime errors per episode
            from rtp.agent.state import EpisodeState

            episode = EpisodeState(instruction=rec["instruction"])
            episode.injected_failures = inj_plan.active()
            print(f"[{tf.name}] ERROR: {type(e).__name__}: {str(e)[:120]}")
        episodes.append(episode)
        print(f"[{tf.name}] '{rec['instruction']}' -> success={episode.success} "
              f"replans={episode.num_replans} injected={episode.injected_failures}")

    recorder.close()
    metrics = compute_metrics(episodes)
    print("\n=== METRICS ===")
    for k, v in metrics.as_dict().items():
        print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

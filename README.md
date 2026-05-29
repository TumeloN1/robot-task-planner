# Robot Task Planner with Self-Correcting Execution

A simulated tabletop manipulation agent. It receives a high-level instruction
(e.g. *"Put the red mug on the tray"*), observes the scene, asks a Gemini LLM to
emit a JSON sequence of tool calls, executes those tools as scripted MuJoCo
primitives, checks postconditions after each step, diagnoses failures, and
**replans without human intervention**.

The interesting research behavior is not generating the plan; it is what happens
when a grasp fails, an object slips, a target is blocked, or perception confuses
two similar objects.

## Status

MVP scaffold. The closed loop runs on **ground-truth perception** (behind a
vision-ready API) with a **custom lightweight MuJoCo scene** (Franka Panda + table
+ objects). No learning yet; episode logs are recorded in a format that seeds
future behavior-cloning / failure-model work.

## Architecture (closed loop)

```
instruction
   -> perceive (SceneState; only visible objects + last-seen memory)
   -> plan (Gemini -> JSON tool calls + rationale)
   -> validate (schema/grounding -> AABB/workspace -> IK feasibility -> loop detection)
   -> for each step: execute primitive -> temporal postcondition check
        on failure -> escalate: rule-based recovery -> LLM replan -> abort
   -> record episode (HDF5 + failure labels)
```

## Design decisions

- **IK**: `mink` is the primary solver (respects joint limits); a small
  damped-least-squares fallback lives in `control/ik.py`. MVP grasps are
  constrained to **top-down** to keep IK robust.
- **Partial observability from day one**: `ObjectState` carries
  `visible` / `confidence` / `last_seen_step`. Occluded objects are hidden, which
  forces the planner to use `find_object` / `look_around` instead of assuming
  global knowledge. This keeps the eventual vision swap from breaking the planner.
- **Temporal checkers**: success must hold for a settle window (~0.5 s sim time),
  not a single frame, so dynamic slips are caught.
- **Anti-loop replanning**: a required `rationale` field, a concise
  `FailureContext` with a `must_change` directive, and validator-side loop
  detection that rejects near-identical replans.

## Setup (macOS)

Requires Python 3.10+.

```bash
cd ~/robot-task-planner
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[dev]"     # or: pip install -r requirements.txt

# Vendor the Franka Panda model (not committed)
python scripts/fetch_assets.py

# Configure the LLM
cp .env.example .env        # then edit .env and set GEMINI_API_KEY
```

Rendering on macOS uses the GLFW backend:

```bash
export MUJOCO_GL=glfw
```

On Linux / Colab use `MUJOCO_GL=egl` for headless rendering.

## Quick start

```bash
python scripts/smoke_mujoco.py        # load the scene and open the viewer
python scripts/smoke_gemini.py        # instruction -> validated plan (no execution)
python scripts/run_episode.py "Put the red mug on the tray"
python scripts/generate_scenes.py     # build a dataset of scenes + tasks
python scripts/evaluate.py            # batch eval -> metrics
```

## Layout

```
configs/      sim / planner / tasks / run YAML configs
assets/       MJCF scene + object snippets; vendored Panda model
src/rtp/      the package: sim, control, perception, primitives,
              planner, agent, tasks, data, eval
scripts/      runnable entrypoints + asset fetch
tests/        unit tests
data/         generated scenes + episode logs (gitignored)
```

## Roadmap

- Vision-based perception (Lab 1 geometry + Open3D, then SAM / GroundingDINO).
- Learned BC primitives and a learned failure model trained on the logged HDF5.
- Colab path for GPU-backed vision/learning (sim stays headless via `MUJOCO_GL=egl`).

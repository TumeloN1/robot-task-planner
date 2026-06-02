"""Run a single instruction end-to-end through the closed loop (live Gemma).

Usage:
    python scripts/run_episode.py "Put the red mug on the tray"
    python scripts/run_episode.py --video data/videos/demo.mp4 "Put the red mug on the tray"
    python scripts/run_episode.py --view "Put the red mug on the tray"
"""

from __future__ import annotations

import argparse
import sys

from _env import planner_kwargs, setup_env

setup_env()

from rtp.agent.build import build_agent  # noqa: E402
from rtp.planner.gemini_client import GeminiPlanner  # noqa: E402
from rtp.sim.observe import PassiveViewerSync, VideoRecorder  # noqa: E402
from rtp.sim.scene_builder import default_scene_spec  # noqa: E402


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "instruction",
        nargs="?",
        default="Put the red mug on the tray",
        help="Natural-language task instruction.",
    )
    parser.add_argument(
        "--video",
        nargs="?",
        const="data/videos/episode.mp4",
        help="Record the executing episode to MP4 (default: data/videos/episode.mp4).",
    )
    parser.add_argument("--view", action="store_true", help="Open the MuJoCo passive viewer.")
    parser.add_argument("--width", type=int, default=1280, help="Video width in pixels.")
    parser.add_argument("--height", type=int, default=720, help="Video height in pixels.")
    parser.add_argument("--fps", type=int, default=30, help="Video frames per second.")
    return parser.parse_args(argv[1:])


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    instruction = args.instruction
    planner = GeminiPlanner(**planner_kwargs())
    ctx, loop = build_agent(default_scene_spec(), planner)

    recorder = None
    viewer = None
    try:
        if args.video:
            recorder = VideoRecorder(args.video, width=args.width, height=args.height, fps=args.fps)
            recorder.attach(ctx.scene)
            print(f"Recording video -> {args.video}")
        if args.view:
            viewer = PassiveViewerSync()
            viewer.attach(ctx.scene)
            print("Opened MuJoCo passive viewer.")

        episode = loop.run(instruction)
    finally:
        if viewer is not None:
            viewer.close(ctx.scene)
        if recorder is not None:
            recorder.close(ctx.scene)

    print(f"\nInstruction: {instruction}")
    print(f"SUCCESS: {episode.success}  replans: {episode.num_replans}  "
          f"steps: {len(episode.steps)}")
    for i, s in enumerate(episode.steps):
        flag = "ok" if s.success else f"FAIL[{s.failure_label}]"
        print(f"  {i:2d} {s.tool}({s.args}) -> {flag}")
    if recorder is not None:
        print(f"Video frames: {recorder.frames} -> {recorder.out_path}")
    return 0 if episode.success else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

"""Sanity check: assemble the scene, settle physics, and render a frame.

Usage:
    python scripts/smoke_mujoco.py            # offscreen render -> data/smoke_scene.png
    python scripts/smoke_mujoco.py --viewer   # interactive viewer (needs a display)
"""

from __future__ import annotations

import sys
from pathlib import Path

from _env import setup_env

setup_env()

from rtp.sim.scene import MuJoCoScene  # noqa: E402
from rtp.sim.scene_builder import default_scene_spec  # noqa: E402


def main(argv: list[str]) -> int:
    scene_spec = default_scene_spec()
    scene = MuJoCoScene.from_scene_spec(scene_spec, timestep=0.002)

    print(f"Model loaded: nq={scene.model.nq} nu={scene.model.nu} nbody={scene.model.nbody}")
    print(f"grip_site (home): {scene.grip_pos().round(3)}")

    # Let objects settle onto the table (~1 s of sim time).
    scene.step(500)

    for obj in scene_spec.objects:
        print(f"  {obj.id:10s} -> {scene.body_pos(obj.id).round(3)}")

    if "--viewer" in argv:
        print("Launching viewer (close the window to exit)...")
        import time

        with scene.launch_viewer() as viewer:
            while viewer.is_running():
                scene.step(1)
                viewer.sync()
                time.sleep(scene.model.opt.timestep)
        return 0

    out = Path("data") / "smoke_scene.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        import imageio.v2 as imageio

        frame = scene.render(width=640, height=480)
        imageio.imwrite(out, frame)
        print(f"Rendered frame -> {out}")
    except Exception as e:  # rendering needs a GL context; physics check still passed
        print(f"Render skipped ({type(e).__name__}: {e}). Physics/model load OK.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

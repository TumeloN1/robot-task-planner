"""Hard-coded pick-and-place proving the control stack end-to-end (no LLM).

Picks the red mug top-down and places it on the tray, printing each stage and a
final success check. Renders before/after frames to data/.

Usage:
    python scripts/smoke_pick_place.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from _env import setup_env

setup_env()

from rtp.control.arm_controller import ArmController  # noqa: E402
from rtp.control.feasibility import check_pose_feasible  # noqa: E402
from rtp.control.gripper import Gripper  # noqa: E402
from rtp.sim.scene import MuJoCoScene  # noqa: E402
from rtp.sim.scene_builder import default_scene_spec  # noqa: E402


def _save(scene, name: str) -> None:
    try:
        import imageio.v2 as imageio

        out = Path("data") / name
        out.parent.mkdir(parents=True, exist_ok=True)
        imageio.imwrite(out, scene.render(width=640, height=480))
        print(f"  rendered -> {out}")
    except Exception as e:
        print(f"  render skipped ({type(e).__name__})")


def main() -> int:
    scene = MuJoCoScene.from_scene_spec(default_scene_spec(), timestep=0.002)
    scene.step(500)  # settle objects
    arm = ArmController(scene)
    grip = Gripper(scene)

    mug = scene.body_pos("red_mug").copy()
    tray = scene.body_pos("tray").copy()
    print(f"mug={mug.round(3)} tray={tray.round(3)}")
    _save(scene, "pp_0_start.png")

    pregrasp = mug + np.array([0, 0, 0.12])
    grasp = mug + np.array([0, 0, 0.02])
    lift = mug + np.array([0, 0, 0.22])
    over_tray = np.array([tray[0], tray[1], 0.65])
    place = np.array([tray[0], tray[1], 0.52])

    feas = check_pose_feasible(scene, pregrasp)
    print(f"pregrasp feasibility: reachable={feas.reachable} collision_free={feas.collision_free}")

    grip.open()
    print(f"move pregrasp: {arm.move_to_pose(pregrasp)}  grip={scene.grip_pos().round(3)}")
    print(f"move grasp:    {arm.move_to_pose(grasp)}  grip={scene.grip_pos().round(3)}")
    grip.close()
    holding = grip.is_holding("red_mug")
    print(f"close gripper -> holding={holding} aperture={grip.finger_aperture():.3f}")

    print(f"lift:          {arm.move_to_pose(lift)}  grip={scene.grip_pos().round(3)}")
    held = grip.is_holding("red_mug")
    print(f"  still holding after lift: {held} mug_z={scene.body_pos('red_mug')[2]:.3f}")
    _save(scene, "pp_1_lifted.png")

    print(f"move over tray:{arm.move_to_pose(over_tray)}  grip={scene.grip_pos().round(3)}")
    print(f"lower to place:{arm.move_to_pose(place)}  grip={scene.grip_pos().round(3)}")
    grip.open()
    arm.settle(1.0)

    mug_f = scene.body_pos("red_mug").copy()
    on_tray = (abs(mug_f[0] - tray[0]) < 0.09 and abs(mug_f[1] - tray[1]) < 0.09
               and mug_f[2] > tray[2] + 0.005)
    print(f"final mug={mug_f.round(3)}  ON TRAY: {on_tray}")
    _save(scene, "pp_2_placed.png")
    return 0 if on_tray else 1


if __name__ == "__main__":
    raise SystemExit(main())

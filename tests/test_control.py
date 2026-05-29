import numpy as np

from rtp.control.arm_controller import ArmController
from rtp.control.feasibility import check_pose_feasible
from rtp.control.gripper import Gripper
from rtp.control.ik import solve_ik
from rtp.sim.scene import MuJoCoScene
from rtp.sim.scene_builder import default_scene_spec


def _scene():
    scene = MuJoCoScene.from_scene_spec(default_scene_spec(), timestep=0.002)
    scene.step(500)
    return scene


def test_ik_reaches_top_down_target():
    scene = _scene()
    mug = scene.body_pos("red_mug")
    target = mug + np.array([0, 0, 0.15])
    q = solve_ik(scene, target)
    assert q is not None and q.shape == (7,)


def test_feasibility_rejects_unreachable_pose():
    scene = _scene()
    far = np.array([2.0, 2.0, 1.5])  # well outside the workspace
    res = check_pose_feasible(scene, far)
    assert res.reachable is False
    assert res.ok is False


def test_pick_and_place_succeeds():
    scene = _scene()
    arm = ArmController(scene)
    grip = Gripper(scene)
    mug = scene.body_pos("red_mug").copy()
    tray = scene.body_pos("tray").copy()

    grip.open()
    assert arm.move_to_pose(mug + np.array([0, 0, 0.12]))
    assert arm.move_to_pose(mug + np.array([0, 0, 0.02]))
    grip.close()
    assert grip.is_holding("red_mug")
    assert arm.move_to_pose(mug + np.array([0, 0, 0.22]))
    assert grip.is_holding("red_mug")  # still held after lift
    arm.move_to_pose(np.array([tray[0], tray[1], 0.65]))
    arm.move_to_pose(np.array([tray[0], tray[1], 0.52]))
    grip.open()
    arm.settle(1.0)

    mug_f = scene.body_pos("red_mug")
    assert abs(mug_f[0] - tray[0]) < 0.09 and abs(mug_f[1] - tray[1]) < 0.09
    assert mug_f[2] > tray[2] + 0.005

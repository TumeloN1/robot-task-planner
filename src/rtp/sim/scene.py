"""MuJoCo scene wrapper.

Owns the `mjModel` / `mjData` pair and exposes a small, stable surface
(step / reset / render / clone_data) used by the rest of the system. Keeps the
model/data separation that MuJoCo encourages so feasibility checks can run on a
cloned `mjData` without touching the live simulation.
"""

from __future__ import annotations

import copy

import mujoco
import numpy as np

from rtp.sim.objects import SceneSpec
from rtp.sim.scene_builder import CAMERA_NAME, GRIP_SITE_NAME, assemble_spec

# Panda home configuration (mirrors the menagerie 'home' keyframe).
ARM_HOME_QPOS: dict[str, float] = {
    "joint1": 0.0,
    "joint2": 0.0,
    "joint3": 0.0,
    "joint4": -1.57079,
    "joint5": 0.0,
    "joint6": 1.57079,
    "joint7": -0.7853,
    "finger_joint1": 0.04,
    "finger_joint2": 0.04,
}
ARM_ACTUATORS = ("actuator1", "actuator2", "actuator3", "actuator4",
                 "actuator5", "actuator6", "actuator7")
GRIPPER_ACTUATOR = "actuator8"
GRIPPER_OPEN_CTRL = 255.0


class MuJoCoScene:
    """Loads a composed model and drives stepping / rendering."""

    def __init__(self, spec: mujoco.MjSpec, *, timestep: float | None = None) -> None:
        self.spec = spec
        self.model = spec.compile()
        if timestep is not None:
            self.model.opt.timestep = timestep
        self.data = mujoco.MjData(self.model)
        self._renderer: mujoco.Renderer | None = None
        self.reset()

    @classmethod
    def from_scene_spec(cls, scene: SceneSpec, *, assets_dir=None,
                        timestep: float | None = None) -> MuJoCoScene:
        kwargs = {} if assets_dir is None else {"assets_dir": assets_dir}
        return cls(assemble_spec(scene, **kwargs), timestep=timestep)

    # -- ids -----------------------------------------------------------------
    def body_id(self, name: str) -> int:
        return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)

    def site_id(self, name: str) -> int:
        return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, name)

    @property
    def grip_site_id(self) -> int:
        return self.site_id(GRIP_SITE_NAME)

    def grip_pos(self) -> np.ndarray:
        return self.data.site_xpos[self.grip_site_id].copy()

    def body_pos(self, name: str) -> np.ndarray:
        return self.data.xpos[self.body_id(name)].copy()

    # -- lifecycle -----------------------------------------------------------
    def reset(self) -> None:
        """Reset to spawn poses (objects) + Panda home (arm), then forward."""
        mujoco.mj_resetData(self.model, self.data)
        for joint, value in ARM_HOME_QPOS.items():
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint)
            if jid >= 0:
                self.data.qpos[self.model.jnt_qposadr[jid]] = value
        # Hold the home pose; open the gripper.
        for i, act in enumerate(ARM_ACTUATORS):
            self._set_ctrl(act, ARM_HOME_QPOS[f"joint{i + 1}"])
        self._set_ctrl(GRIPPER_ACTUATOR, GRIPPER_OPEN_CTRL)
        mujoco.mj_forward(self.model, self.data)

    def _set_ctrl(self, actuator: str, value: float) -> None:
        aid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator)
        if aid >= 0:
            self.data.ctrl[aid] = value

    def step(self, n: int = 1) -> None:
        for _ in range(n):
            mujoco.mj_step(self.model, self.data)

    def clone_data(self) -> mujoco.MjData:
        """Deep copy of mjData for off-line feasibility checks."""
        return copy.deepcopy(self.data)

    # -- rendering -----------------------------------------------------------
    def render(self, *, width: int = 640, height: int = 480,
               camera: str = CAMERA_NAME) -> np.ndarray:
        """Return an offscreen RGB frame (H, W, 3) uint8."""
        if self._renderer is None or self._renderer.width != width \
                or self._renderer.height != height:
            self._renderer = mujoco.Renderer(self.model, height=height, width=width)
        self._renderer.update_scene(self.data, camera=camera)
        return self._renderer.render()

    def launch_viewer(self):
        """Launch the interactive passive viewer (requires a display)."""
        import mujoco.viewer

        return mujoco.viewer.launch_passive(self.model, self.data)

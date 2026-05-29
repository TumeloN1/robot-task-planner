"""Assemble a ready-to-run agent (scene + controllers + perception + loop)."""

from __future__ import annotations

from rtp.agent.context import ExecutionContext
from rtp.agent.loop import AgentConfig, AgentLoop
from rtp.control.arm_controller import ArmController
from rtp.control.gripper import Gripper
from rtp.perception.ground_truth import GroundTruthPerception
from rtp.sim.objects import SceneSpec
from rtp.sim.scene import MuJoCoScene


def build_agent(scene_spec: SceneSpec, planner, *, injector=None, recorder=None,
                config: AgentConfig | None = None, timestep: float = 0.002,
                settle_steps: int = 500):
    """Build the execution context and agent loop for a scene.

    Returns (ctx, loop).
    """
    scene = MuJoCoScene.from_scene_spec(scene_spec, timestep=timestep)
    scene.step(settle_steps)  # let objects settle onto the table
    arm = ArmController(scene)
    gripper = Gripper(scene)
    perception = GroundTruthPerception(scene, scene_spec)
    ctx = ExecutionContext(scene, arm, gripper, perception, injector=injector)
    loop = AgentLoop(ctx, planner, recorder=recorder, config=config)
    return ctx, loop

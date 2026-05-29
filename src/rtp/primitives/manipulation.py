"""Manipulation primitives: grasp and release."""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel, Field

from rtp.agent.context import GRASP_OFFSET
from rtp.primitives.registry import PrimitiveResult, primitive


class GraspArgs(BaseModel):
    object: str = Field(description="Name/id of the object to grasp (top-down).")


class ReleaseArgs(BaseModel):
    pass


@primitive("grasp", GraspArgs,
           description="Descend and close the gripper on the object using a top-down grasp.")
def grasp(ctx, args: GraspArgs) -> PrimitiveResult:
    obj = ctx.resolve(args.object)
    if obj is None:
        return PrimitiveResult(False, message=f"unknown object {args.object!r}")
    target = ctx.object_pos(obj.id) + np.array([0, 0, GRASP_OFFSET])
    ctx.gripper.open()
    if not ctx.arm.move_to_pose(target):
        return PrimitiveResult(False, message="grasp pose unreachable")
    ctx.gripper.close()
    holding = ctx.gripper.is_holding(obj.id)
    if holding:
        ctx.held_object = obj.id
    return PrimitiveResult(holding, info={"object": obj.id},
                           message="grasped" if holding else "grasp did not secure object")


@primitive("release", ReleaseArgs,
           description="Open the gripper to release the held object.")
def release(ctx, args: ReleaseArgs) -> PrimitiveResult:
    ctx.gripper.open()
    ctx.arm.settle(0.5)
    ctx.held_object = None
    return PrimitiveResult(True, message="released")

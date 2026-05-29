"""Motion primitives: pregrasp approach and free-space pose moves."""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel, Field

from rtp.agent.context import (
    CARRY_Z,
    PLACE_APPROACH_Z,
    PLACE_Z,
    PREGRASP_OFFSET,
)
from rtp.primitives.registry import PrimitiveResult, primitive


class MoveToPregraspArgs(BaseModel):
    object: str = Field(description="Name/id of the object to approach.")


class MoveToPoseArgs(BaseModel):
    target: str = Field(description="Target object/location to move (the held object) over.")


@primitive("move_to_pregrasp", MoveToPregraspArgs,
           description="Move the gripper to a top-down pregrasp pose above the object.")
def move_to_pregrasp(ctx, args: MoveToPregraspArgs) -> PrimitiveResult:
    obj = ctx.resolve(args.object)
    if obj is None:
        return PrimitiveResult(False, message=f"unknown object {args.object!r}")
    target = ctx.object_pos(obj.id) + np.array([0, 0, PREGRASP_OFFSET])
    ok = ctx.arm.move_to_pose(target)
    return PrimitiveResult(ok, message="at pregrasp" if ok else "pregrasp unreachable")


@primitive("move_to_pose", MoveToPoseArgs,
           description="Move the held object over the target location, ready to release.")
def move_to_pose(ctx, args: MoveToPoseArgs) -> PrimitiveResult:
    obj = ctx.resolve(args.target)
    if obj is None:
        return PrimitiveResult(False, message=f"unknown target {args.target!r}")
    tx, ty = ctx.object_pos(obj.id)[:2]

    # Lift first so the held object clears obstacles, then carry and lower.
    if ctx.held_object is not None:
        ctx.arm.move_to_pose(np.array([*ctx.object_pos(ctx.held_object)[:2], CARRY_Z]))
    if not ctx.arm.move_to_pose(np.array([tx, ty, PLACE_APPROACH_Z])):
        return PrimitiveResult(False, message="approach pose unreachable")
    ok = ctx.arm.move_to_pose(np.array([tx, ty, PLACE_Z]))

    # Injected slip: the object may be dropped mid-transport.
    if ok and ctx.held_object and ctx.injector is not None:
        if ctx.injector.maybe_slip(ctx.scene, ctx.held_object):
            ctx.gripper.open()
            ctx.held_object = None
            return PrimitiveResult(
                False, info={"slipped": True}, message="object slipped during transport")

    return PrimitiveResult(ok, message="over target" if ok else "place pose unreachable")

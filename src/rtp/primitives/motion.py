"""Motion primitives: pregrasp approach and free-space pose moves."""

from __future__ import annotations

from pydantic import BaseModel, Field

from rtp.primitives.registry import PrimitiveResult, primitive


class MoveToPregraspArgs(BaseModel):
    object: str = Field(description="Name/id of the object to approach.")


class MoveToPoseArgs(BaseModel):
    target: str = Field(description="Named target pose, e.g. 'tray_place_pose'.")


@primitive("move_to_pregrasp", MoveToPregraspArgs,
           description="Move the gripper to a top-down pregrasp pose above the object.")
def move_to_pregrasp(ctx, args: MoveToPregraspArgs) -> PrimitiveResult:
    raise NotImplementedError


@primitive("move_to_pose", MoveToPoseArgs,
           description="Move the (possibly holding) gripper to a named target pose.")
def move_to_pose(ctx, args: MoveToPoseArgs) -> PrimitiveResult:
    raise NotImplementedError

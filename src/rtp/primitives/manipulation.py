"""Manipulation primitives: grasp and release."""

from __future__ import annotations

from pydantic import BaseModel, Field

from rtp.primitives.registry import PrimitiveResult, primitive


class GraspArgs(BaseModel):
    object: str = Field(description="Name/id of the object to grasp (top-down).")


class ReleaseArgs(BaseModel):
    pass


@primitive("grasp", GraspArgs,
           description="Close the gripper on the object using a top-down grasp.")
def grasp(ctx, args: GraspArgs) -> PrimitiveResult:
    raise NotImplementedError


@primitive("release", ReleaseArgs,
           description="Open the gripper to release the held object.")
def release(ctx, args: ReleaseArgs) -> PrimitiveResult:
    raise NotImplementedError

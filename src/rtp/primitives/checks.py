"""Postcondition checkers (temporal).

Checkers must verify a condition holds across a settle window of N consecutive
sim steps, not a single frame, so a grasp that slips shortly after looking
successful is correctly reported as a failure.
"""

from __future__ import annotations

from typing import Callable

from pydantic import BaseModel, Field

from rtp.primitives.registry import PrimitiveResult, primitive


def stable_for(scene, predicate: Callable[[], bool], *, seconds: float = 0.5) -> bool:
    """True only if `predicate` holds for every step in the settle window."""
    raise NotImplementedError


class CheckGraspArgs(BaseModel):
    object: str = Field(description="Object that should be held after the grasp.")


class CheckTaskArgs(BaseModel):
    object: str
    relation: str = Field(description="Spatial relation, e.g. 'on', 'near', 'left_of'.")
    target: str


@primitive("check_grasp_success", CheckGraspArgs, is_check=True,
           description="Verify the object is held stably for the settle window.")
def check_grasp_success(ctx, args: CheckGraspArgs) -> PrimitiveResult:
    raise NotImplementedError


@primitive("check_task_success", CheckTaskArgs, is_check=True,
           description="Verify the goal relation holds stably for the settle window.")
def check_task_success(ctx, args: CheckTaskArgs) -> PrimitiveResult:
    raise NotImplementedError

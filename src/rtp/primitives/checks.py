"""Postcondition checkers (temporal).

Checkers verify a condition holds across a settle window of several consecutive
sim steps, not a single frame, so a grasp that slips shortly after looking
successful is correctly reported as a failure.
"""

from __future__ import annotations

from collections.abc import Callable

from pydantic import BaseModel, Field

from rtp.primitives.registry import PrimitiveResult, primitive
from rtp.tasks import predicates


def stable_for(scene, predicate: Callable[[], bool], *, seconds: float = 0.5,
               samples: int = 5) -> bool:
    """True only if `predicate` holds at every sample across the settle window."""
    step_chunk = max(1, int((seconds / samples) / scene.model.opt.timestep))
    for _ in range(samples):
        scene.step(step_chunk)
        if not predicate():
            return False
    return True


class CheckGraspArgs(BaseModel):
    object: str = Field(description="Object that should be held after the grasp.")


class CheckTaskArgs(BaseModel):
    object: str
    relation: str = Field(description="Spatial relation, e.g. 'on', 'near', 'left_of'.")
    target: str


@primitive("check_grasp_success", CheckGraspArgs, is_check=True,
           description="Verify the object is held stably for the settle window.")
def check_grasp_success(ctx, args: CheckGraspArgs) -> PrimitiveResult:
    obj = ctx.resolve(args.object)
    if obj is None:
        return PrimitiveResult(False, message=f"unknown object {args.object!r}")
    held = stable_for(ctx.scene, lambda: ctx.gripper.is_holding(obj.id))
    return PrimitiveResult(held, message="grasp stable" if held else "object not held stably")


@primitive("check_task_success", CheckTaskArgs, is_check=True,
           description="Verify the goal relation holds stably for the settle window.")
def check_task_success(ctx, args: CheckTaskArgs) -> PrimitiveResult:
    def holds() -> bool:
        ctx.perceive()
        subj = ctx.resolve(args.object)
        tgt = ctx.resolve(args.target)
        if subj is None or tgt is None:
            return False
        try:
            return predicates.evaluate(args.relation, subj, tgt)
        except ValueError:
            return False

    ok = stable_for(ctx.scene, holds)
    return PrimitiveResult(ok, info={"relation": args.relation},
                           message="task satisfied" if ok else "goal relation not satisfied")

"""Query primitives: locate objects, search under partial observability."""

from __future__ import annotations

from pydantic import BaseModel, Field

from rtp.primitives.registry import PrimitiveResult, primitive


class FindObjectArgs(BaseModel):
    query: str = Field(description="Natural-language object description, e.g. 'red mug'.")


class LookAroundArgs(BaseModel):
    pass


@primitive("find_object", FindObjectArgs,
           description="Locate an object matching the query in the current scene.")
def find_object(ctx, args: FindObjectArgs) -> PrimitiveResult:
    ctx.perceive()
    obj = ctx.resolve(args.query)
    if obj is None:
        return PrimitiveResult(False, message=f"object matching {args.query!r} not visible")
    return PrimitiveResult(True, info={"object": obj.id, "position": obj.pose.position.tolist()},
                           message=f"found {obj.name}")


@primitive("look_around", LookAroundArgs,
           description="Re-scan the scene to reveal objects that may be occluded.")
def look_around(ctx, args: LookAroundArgs) -> PrimitiveResult:
    scene_state = ctx.perceive()
    names = [o.name for o in scene_state.visible_objects]
    return PrimitiveResult(True, info={"visible": names}, message=f"visible: {', '.join(names)}")

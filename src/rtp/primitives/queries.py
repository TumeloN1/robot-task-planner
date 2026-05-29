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
    raise NotImplementedError


@primitive("look_around", LookAroundArgs,
           description="Scan the scene to reveal objects that are currently occluded.")
def look_around(ctx, args: LookAroundArgs) -> PrimitiveResult:
    raise NotImplementedError

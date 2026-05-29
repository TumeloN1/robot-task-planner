"""Scripted low-level primitives exposed to the planner as JSON tools.

`registry` is the single source of truth for tool names + argument models; the
planner schema and the executor both derive from it so they never drift.
"""

from rtp.primitives.registry import PRIMITIVES, PrimitiveResult, primitive

__all__ = ["PRIMITIVES", "PrimitiveResult", "primitive"]

"""Scripted low-level primitives exposed to the planner as JSON tools.

Importing this package registers every primitive in the `PRIMITIVES` table.
`registry` is the single source of truth for tool names + argument models; the
planner schema and the executor both derive from it so they never drift.
"""

# Import submodules for their registration side effects.
from rtp.primitives import checks, manipulation, motion, queries  # noqa: E402,F401
from rtp.primitives.registry import PRIMITIVES, PrimitiveResult, primitive

__all__ = ["PRIMITIVES", "PrimitiveResult", "primitive"]

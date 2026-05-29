"""Primitive registry: maps tool name -> callable + Pydantic argument model.

This is the single source of truth. `planner/schema.py` derives the Gemini JSON
schema from `PRIMITIVES`, and `agent/loop.py` dispatches execution through it,
so the prompt, the validator, and the executor can never disagree about the
available tools or their arguments.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from pydantic import BaseModel


@dataclass
class PrimitiveResult:
    """Outcome of executing one primitive."""

    success: bool
    info: dict = field(default_factory=dict)
    message: str = ""


@dataclass
class PrimitiveSpec:
    name: str
    args_model: type[BaseModel]
    fn: Callable
    description: str
    is_check: bool = False  # postcondition checkers vs. action primitives


PRIMITIVES: dict[str, PrimitiveSpec] = {}


def primitive(name: str, args_model: type[BaseModel], *, description: str,
              is_check: bool = False) -> Callable:
    """Decorator registering a primitive in the global PRIMITIVES table."""

    def wrap(fn: Callable) -> Callable:
        PRIMITIVES[name] = PrimitiveSpec(
            name=name, args_model=args_model, fn=fn,
            description=description, is_check=is_check,
        )
        return fn

    return wrap

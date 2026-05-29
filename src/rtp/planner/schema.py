"""Structured-output schema for the planner.

The `Plan` requires a `rationale` field that the model fills *before* the tool
list. On a replan this forces the model to state what it is changing and why,
which is the primary defense against re-emitting a previously failed plan.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class ToolCall(BaseModel):
    tool: str = Field(description="Registered primitive name.")
    args: dict = Field(default_factory=dict, description="Arguments for the tool.")


class Plan(BaseModel):
    rationale: str = Field(
        description="Brief reasoning. On a replan, state explicitly what changed "
                    "from the previous failed attempt and why."
    )
    tool_calls: list[ToolCall] = Field(description="Ordered tool calls to execute.")


def build_response_json_schema() -> dict:
    """JSON schema passed to Gemini (`response_json_schema`).

    Derived from the primitive registry so the catalog of valid tools stays in
    sync with what the executor can run.
    """
    return Plan.model_json_schema()

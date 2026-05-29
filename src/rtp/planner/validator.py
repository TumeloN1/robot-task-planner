"""Deterministic plan validation, cheapest checks first.

1. Schema/grounding : tool exists, args match the model, object refs resolve
                      (either visible now or located by an earlier find_object).
2. IK feasibility   : grasp/approach poses are reachable + collision-free,
                      checked on a cloned mjData (only when a scene context is
                      supplied).
3. Loop detection   : reject a replan identical to one that already failed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from rtp.perception.api import SceneState
from rtp.planner.schema import Plan
from rtp.primitives.registry import PRIMITIVES

# Tool arg fields that name an object/target the plan must be able to ground.
_OBJECT_ARGS = {
    "move_to_pregrasp": ["object"],
    "grasp": ["object"],
    "move_to_pose": ["target"],
    "check_grasp_success": ["object"],
    "check_task_success": ["object", "target"],
}


@dataclass
class ValidationResult:
    valid: bool
    errors: list[str] = field(default_factory=list)

    def as_repair_text(self) -> str:
        return "; ".join(self.errors)


def _norm(s: str) -> str:
    return s.strip().lower().replace("_", " ")


def _matches(query: str, name: str) -> bool:
    return all(tok in _norm(name) for tok in _norm(query).split())


def _plan_signature(plan: Plan) -> tuple:
    return tuple((c.tool, tuple(sorted((str(k), str(v)) for k, v in c.args.items())))
                 for c in plan.tool_calls)


def validate_plan(plan: Plan, scene: SceneState, *, scene_ctx=None,
                  failed_plans: list[Plan] | None = None) -> ValidationResult:
    errors: list[str] = []

    if not plan.tool_calls:
        errors.append("plan has no tool_calls")

    # Names/ids groundable now, plus anything an earlier find_object will locate.
    known = [o.name for o in scene.all_known()] + [o.id for o in scene.all_known()]
    known += [c.args.get("query", "") for c in plan.tool_calls if c.tool == "find_object"]

    for i, call in enumerate(plan.tool_calls):
        spec = PRIMITIVES.get(call.tool)
        if spec is None:
            errors.append(f"step {i}: unknown tool {call.tool!r}")
            continue
        try:
            spec.args_model(**call.args)
        except Exception as e:
            errors.append(f"step {i}: bad args for {call.tool} ({e})")
            continue
        for fld in _OBJECT_ARGS.get(call.tool, []):
            ref = call.args.get(fld, "")
            if ref and not any(_matches(ref, name) for name in known if name):
                errors.append(f"step {i}: {call.tool} references ungroundable {fld}={ref!r}")

    # IK feasibility for grasp/approach steps (precise, on a cloned mjData).
    if scene_ctx is not None and not errors:
        errors += _feasibility_errors(plan, scene_ctx)

    # Loop detection.
    if failed_plans:
        sig = _plan_signature(plan)
        if any(_plan_signature(fp) == sig for fp in failed_plans):
            errors.append("plan is identical to a previously failed attempt; change the approach")

    return ValidationResult(valid=not errors, errors=errors)


def _feasibility_errors(plan: Plan, ctx) -> list[str]:
    import numpy as np

    from rtp.agent.context import GRASP_OFFSET, PREGRASP_OFFSET
    from rtp.control.feasibility import check_pose_feasible

    errors: list[str] = []
    offsets = {"move_to_pregrasp": PREGRASP_OFFSET, "grasp": GRASP_OFFSET}
    for i, call in enumerate(plan.tool_calls):
        if call.tool not in offsets:
            continue
        obj = ctx.resolve(call.args.get("object", ""))
        if obj is None:
            continue  # grounding handled above
        target = ctx.object_pos(obj.id) + np.array([0, 0, offsets[call.tool]])
        res = check_pose_feasible(ctx.scene, target)
        if not res.ok:
            errors.append(f"step {i}: {call.tool} pose infeasible ({res.reason})")
    return errors

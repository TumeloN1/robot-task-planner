"""The closed-loop orchestrator: the heart of the system.

perceive -> plan -> validate -> execute each step + temporal check
   on failure, escalate:
     (1) cheap rule-based recovery (one immediate grasp retry),
     (2) LLM replan with a concise FailureContext,
     (3) abort.
Caps total replans; validator loop-detection blocks identical retries.
Every step is recorded.
"""

from __future__ import annotations

from dataclasses import dataclass

import rtp.primitives  # noqa: F401  (populates the PRIMITIVES registry)
from rtp.agent.diagnosis import FailureContext, classify
from rtp.agent.state import EpisodeState, StepRecord
from rtp.planner.validator import validate_plan
from rtp.primitives.registry import PRIMITIVES


@dataclass
class AgentConfig:
    max_steps: int = 30
    max_replans: int = 3


class AgentLoop:
    def __init__(self, ctx, planner, *, recorder=None, config: AgentConfig | None = None) -> None:
        self.ctx = ctx
        self.planner = planner
        self.recorder = recorder
        self.config = config or AgentConfig()

    def run(self, instruction: str, *, seed: int = 0) -> EpisodeState:
        ctx = self.ctx
        episode = EpisodeState(instruction=instruction)
        if self.recorder:
            self.recorder.start_episode(instruction, seed)

        ctx.perceive()
        plan = self.planner.propose(instruction, ctx.scene_state)
        replans = 0

        while True:
            vres = validate_plan(plan, ctx.scene_state, scene_ctx=ctx,
                                 failed_plans=episode.failed_plans)
            if not vres.valid:
                fctx = FailureContext(
                    failure_type=classify("validator", _Reason(vres.as_repair_text())),
                    attempted="proposed plan", result=vres.as_repair_text(),
                    prior_attempts=replans,
                )
                if not self._can_replan(episode, plan, replans):
                    break
                replans += 1
                episode.num_replans = replans
                ctx.perceive()
                plan = self.planner.propose(instruction, ctx.scene_state, fctx.render())
                continue

            failure = self._execute_plan(plan, episode)
            if failure is None:
                episode.success = True
                break
            if not self._can_replan(episode, plan, replans):
                break
            replans += 1
            episode.num_replans = replans
            failure.prior_attempts = replans
            ctx.perceive()
            plan = self.planner.propose(instruction, ctx.scene_state, failure.render())

        if self.recorder:
            self.recorder.end_episode(success=episode.success, num_replans=episode.num_replans,
                                      injected_failures=episode.injected_failures)
        return episode

    def _can_replan(self, episode: EpisodeState, plan, replans: int) -> bool:
        episode.failed_plans.append(plan)
        return replans < self.config.max_replans

    def _execute_plan(self, plan, episode: EpisodeState) -> FailureContext | None:
        ctx = self.ctx
        for call in plan.tool_calls:
            if len(episode.steps) >= self.config.max_steps:
                return FailureContext(classify("timeout", _Reason("step budget exhausted")),
                                      attempted="plan execution", result="step budget exhausted")
            spec = PRIMITIVES[call.tool]
            args = spec.args_model(**call.args)
            result = spec.fn(ctx, args)

            # Rule-based recovery rung: one immediate grasp retry.
            if call.tool == "grasp" and not result.success:
                result = spec.fn(ctx, args)

            failure_label = None
            if not result.success:
                ftype = classify(call.tool, result)
                failure_label = ftype.value
            self._record(episode, call, result, spec.is_check, failure_label)

            if not result.success:
                return FailureContext(
                    failure_type=classify(call.tool, result),
                    attempted=f"{call.tool}({call.args})",
                    result=result.message,
                )
        return None

    def _record(self, episode, call, result, is_check, failure_label) -> None:
        rec = StepRecord(
            tool=call.tool, args=dict(call.args), success=result.success,
            postcondition_pass=result.success if is_check else None,
            failure_label=failure_label, info=dict(result.info),
        )
        episode.steps.append(rec)
        if self.recorder:
            self.recorder.log_step(rec)


class _Reason:
    """Minimal stand-in so classify() can read .message/.info for non-primitive failures."""

    def __init__(self, message: str) -> None:
        self.message = message
        self.info: dict = {}

"""The closed-loop orchestrator: the heart of the system.

perceive -> plan -> validate -> execute each step + temporal check
   on failure, escalate:
     (1) rule-based recovery,
     (2) LLM replan with FailureContext,
     (3) abort.
Caps total replans and steps; validator loop-detection blocks identical retries.
Every step is recorded.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AgentConfig:
    max_steps: int = 30
    max_replans: int = 3


class AgentLoop:
    def __init__(self, *, scene, perception, planner, validator, executor,
                 recorder=None, config: AgentConfig | None = None) -> None:
        self.scene = scene
        self.perception = perception
        self.planner = planner
        self.validator = validator
        self.executor = executor
        self.recorder = recorder
        self.config = config or AgentConfig()

    def run(self, instruction: str):
        """Run one episode to success/abort; return the EpisodeState.

        TODO: implement the perceive/plan/validate/execute/diagnose/replan loop
        with the escalation ladder described in the module docstring.
        """
        raise NotImplementedError

"""Evaluation metrics aggregated over a batch of episodes.

Headline metrics:
  - task_success_rate
  - recovery_success_rate  : success | an injected failure occurred
  - plan_validity_rate     : plan valid on first try (no repair)
  - mean_replans
  - mean_steps_to_success
Recovery rate is also broken down per FailureType - the key research output.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Metrics:
    task_success_rate: float = 0.0
    recovery_success_rate: float = 0.0
    plan_validity_rate: float = 0.0
    mean_replans: float = 0.0
    mean_steps_to_success: float = 0.0
    recovery_by_failure_type: dict[str, float] = field(default_factory=dict)


def compute_metrics(episodes: list) -> Metrics:
    """Aggregate metrics from a list of EpisodeState (or logged records)."""
    raise NotImplementedError

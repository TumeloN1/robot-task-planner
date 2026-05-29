"""Evaluation metrics aggregated over a batch of episodes.

Headline metrics:
  - task_success_rate
  - recovery_success_rate  : success | an injected failure occurred
  - plan_validity_rate     : proxy = fraction needing no replan
  - mean_replans
  - mean_steps_to_success
Recovery rate is also broken down per injected failure type - the key research
output.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Metrics:
    n_episodes: int = 0
    task_success_rate: float = 0.0
    recovery_success_rate: float = 0.0
    plan_validity_rate: float = 0.0
    mean_replans: float = 0.0
    mean_steps_to_success: float = 0.0
    recovery_by_failure_type: dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return self.__dict__


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def compute_metrics(episodes: list) -> Metrics:
    """Aggregate metrics from a list of EpisodeState."""
    if not episodes:
        return Metrics()

    successes = [bool(e.success) for e in episodes]
    injected = [e for e in episodes if e.injected_failures]
    success_steps = [len(e.steps) for e in episodes if e.success]

    by_type: dict[str, list[bool]] = {}
    for e in episodes:
        for f in e.injected_failures:
            by_type.setdefault(f, []).append(bool(e.success))

    return Metrics(
        n_episodes=len(episodes),
        task_success_rate=_mean([float(s) for s in successes]),
        recovery_success_rate=_mean([float(e.success) for e in injected]),
        plan_validity_rate=_mean([float(e.num_replans == 0) for e in episodes]),
        mean_replans=_mean([float(e.num_replans) for e in episodes]),
        mean_steps_to_success=_mean([float(s) for s in success_steps]),
        recovery_by_failure_type={k: _mean([float(v) for v in vs]) for k, vs in by_type.items()},
    )

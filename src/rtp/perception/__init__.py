"""Perception layer.

The `api` module defines the vision-ready contract (`ObjectState`, `SceneState`,
`PerceptionProvider`). `ground_truth` fills it from the simulator; `noise` wraps
any provider to inject perception failures.
"""

from rtp.perception.api import ObjectState, PerceptionProvider, SceneState

__all__ = ["ObjectState", "SceneState", "PerceptionProvider"]

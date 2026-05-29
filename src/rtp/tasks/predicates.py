"""Deterministic spatial relations used as success criteria.

Evaluated from object poses/AABBs (never by the LLM) so task success is
objective and reproducible.
"""

from __future__ import annotations

from rtp.perception.api import ObjectState


def on(subject: ObjectState, target: ObjectState) -> bool:
    """True if `subject` rests on top of `target` (xy overlap + z just above)."""
    raise NotImplementedError


def near(subject: ObjectState, target: ObjectState, threshold_m: float = 0.12) -> bool:
    """True if centers are within `threshold_m`."""
    raise NotImplementedError


def left_of(subject: ObjectState, target: ObjectState) -> bool:
    """True if `subject` is to the left of `target` in the world frame."""
    raise NotImplementedError


def inside(subject: ObjectState, target: ObjectState) -> bool:
    """True if `subject` is within the footprint of `target` (e.g. a tray)."""
    raise NotImplementedError


PREDICATES = {"on": on, "near": near, "left_of": left_of, "inside": inside}

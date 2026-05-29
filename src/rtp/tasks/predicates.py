"""Deterministic spatial relations used as success criteria.

Evaluated from object poses/AABBs (never by the LLM) so task success is
objective and reproducible. World frame: the robot base is at the origin and the
table extends in +x; +y is the robot's left.
"""

from __future__ import annotations

import numpy as np

from rtp.perception.api import ObjectState


def _xy(o: ObjectState) -> np.ndarray:
    return o.pose.position[:2]


def _footprint_half(o: ObjectState) -> np.ndarray:
    return (o.aabb[1, :2] - o.aabb[0, :2]) / 2.0


def on(subject: ObjectState, target: ObjectState) -> bool:
    """True if `subject` rests on top of `target` (xy within footprint, z above)."""
    half = _footprint_half(target)
    within_xy = bool(np.all(np.abs(_xy(subject) - _xy(target)) <= half))
    above = subject.pose.position[2] > target.pose.position[2]
    return within_xy and above


def near(subject: ObjectState, target: ObjectState, threshold_m: float = 0.12) -> bool:
    """True if horizontal centers are within `threshold_m`."""
    return float(np.linalg.norm(_xy(subject) - _xy(target))) < threshold_m


def left_of(subject: ObjectState, target: ObjectState, margin_m: float = 0.02) -> bool:
    """True if `subject` is to the robot's left (greater +y) of `target`."""
    return subject.pose.position[1] > target.pose.position[1] + margin_m


def inside(subject: ObjectState, target: ObjectState) -> bool:
    """True if `subject`'s center is within the footprint of `target`."""
    half = _footprint_half(target)
    return bool(np.all(np.abs(_xy(subject) - _xy(target)) <= half))


PREDICATES = {"on": on, "near": near, "left_of": left_of, "inside": inside}


def evaluate(relation: str, subject: ObjectState, target: ObjectState, **kwargs) -> bool:
    """Look up and evaluate a relation by name."""
    fn = PREDICATES.get(relation)
    if fn is None:
        raise ValueError(f"Unknown relation: {relation!r}")
    return fn(subject, target, **kwargs)

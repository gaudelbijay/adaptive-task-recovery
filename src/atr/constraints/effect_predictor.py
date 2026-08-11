"""Conservative geometric side-effect prediction for the intent guard.

D-083 made predicted affected objects part of the guard interface. This module
provides the first producer: objects whose privileged-state centers lie within
a clearance radius of a planned straight-line motion segment. It is a cheap
high-level screening model, not collision-accurate robot geometry.

D-085 extends the same check across every segment of a waypoint path, avoiding
both missed effects on later legs and false positives caused by replacing a
bent path with its direct start-to-end chord.

D-086 optionally expands the corridor by each object's collision radius. The
default radius is zero for backward-compatible point-center screening.
"""

from __future__ import annotations

import numpy as np

from atr.feasibility.oracle import WorldState


def _point_segment_distance(point: np.ndarray, start: np.ndarray, end: np.ndarray) -> float:
    segment = end - start
    length_squared = float(np.dot(segment, segment))
    if length_squared == 0.0:
        return float(np.linalg.norm(point - start))
    fraction = float(np.dot(point - start, segment) / length_squared)
    closest = start + min(1.0, max(0.0, fraction)) * segment
    return float(np.linalg.norm(point - closest))


def predict_affected_objects(
    state: WorldState,
    start_position,
    end_position,
    clearance_radius: float,
    exclude_objects: frozenset[str] = frozenset(),
    object_radii: dict[str, float] | None = None,
) -> frozenset[str]:
    """Return existing object centers within a swept straight-line corridor.

    Callers normally exclude the intended target because `validate_action()`
    already treats it as an implicit effect. Missing/destroyed objects are not
    physical obstacles and are ignored.
    """
    return predict_affected_objects_along_path(
        state,
        (start_position, end_position),
        clearance_radius,
        exclude_objects=exclude_objects,
        object_radii=object_radii,
    )


def predict_affected_objects_along_path(
    state: WorldState,
    waypoints,
    clearance_radius: float,
    exclude_objects: frozenset[str] = frozenset(),
    object_radii: dict[str, float] | None = None,
) -> frozenset[str]:
    """Return objects whose optional radius overlaps any waypoint segment.

    Consecutive duplicate waypoints are valid and become spherical clearance
    checks. A path needs at least two xyz waypoints so an empty motion cannot be
    mistaken for a fully checked trajectory.
    """
    if clearance_radius < 0:
        raise ValueError("clearance_radius must be non-negative")
    radii = object_radii or {}
    if any(radius < 0 for radius in radii.values()):
        raise ValueError("object radii must be non-negative")
    points = tuple(np.asarray(point, dtype=float) for point in waypoints)
    if len(points) < 2:
        raise ValueError("waypoints must contain at least two xyz vectors")
    if any(point.shape != (3,) for point in points):
        raise ValueError("every waypoint must be an xyz vector")
    segments = tuple(zip(points, points[1:]))
    affected = set()
    for object_id, object_state in state.items():
        if object_id in exclude_objects or not object_state.exists or object_state.position is None:
            continue
        point = np.asarray(object_state.position, dtype=float)
        if point.shape != (3,):
            raise ValueError(f"position for {object_id!r} must be an xyz vector")
        if any(
            _point_segment_distance(point, start, end)
            <= clearance_radius + radii.get(object_id, 0.0)
            for start, end in segments
        ):
            affected.add(object_id)
    return frozenset(affected)

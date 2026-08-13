"""D-101: mobile-base path screening uses planar, not 3D, clearance."""

import numpy as np

from atr.envs.navigation import screen_navigation_path
from atr.feasibility.oracle import ObjectState
from atr.language.goal_graph import canonical_example


def _object(position):
    return ObjectState(exists=True, position=np.asarray(position, dtype=float))


def _state(protected_position):
    return {
        "red_mug": _object((1.0, 1.0, 0.7)),
        "glass": _object(protected_position),
        "blue_bowl": _object((2.0, 2.0, 0.7)),
        "medicine_bottle": _object((-1.0, -1.0, 0.7)),
    }


def test_floor_level_protected_object_in_xy_corridor_is_detected():
    allowed, reason, effects = screen_navigation_path(
        [(0.0, 0.0), (1.0, 0.0)],
        "red_mug",
        canonical_example(),
        _state((0.5, 0.02, 0.05)),
        travel_height=0.5,
        clearance_radius=0.05,
    )

    assert effects == frozenset({"glass"})
    assert allowed is False
    assert "dont_move_glass" in reason


def test_navigation_effects_are_invariant_to_object_center_height():
    outcomes = []
    for height in (0.05, 0.5, 1.2):
        allowed, _, effects = screen_navigation_path(
            [(0.0, 0.0), (1.0, 0.0)],
            "red_mug",
            canonical_example(),
            _state((0.5, 0.02, height)),
            travel_height=0.5,
            clearance_radius=0.05,
        )
        outcomes.append((allowed, effects))

    assert outcomes == [(False, frozenset({"glass"}))] * 3


def test_planar_projection_does_not_flag_object_outside_xy_clearance():
    allowed, _, effects = screen_navigation_path(
        [(0.0, 0.0), (1.0, 0.0)],
        "red_mug",
        canonical_example(),
        _state((0.5, 0.2, 0.05)),
        travel_height=0.5,
        clearance_radius=0.05,
    )

    assert effects == frozenset()
    assert allowed is True

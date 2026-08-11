"""D-084: geometric producer for D-083's predicted action effects."""

import numpy as np
import pytest

from atr.constraints.effect_predictor import (
    predict_affected_objects,
    predict_affected_objects_along_path,
)
from atr.constraints.intent_guard import validate_action
from atr.feasibility.oracle import ObjectState
from atr.envs.navigation import screen_navigation_path
from atr.language.goal_graph import canonical_example


def _object(position, exists=True):
    return ObjectState(exists=exists, position=np.asarray(position, dtype=float) if exists else None)


class TestSweptCorridorEffects:
    def test_flags_object_near_middle_of_motion_not_only_endpoints(self):
        state = {"glass": _object((0.5, 0.04, 0.0))}
        assert predict_affected_objects(state, (0, 0, 0), (1, 0, 0), 0.05) == frozenset({"glass"})

    def test_ignores_objects_beyond_clearance_and_destroyed_objects(self):
        state = {
            "far": _object((0.5, 0.2, 0.0)),
            "destroyed": _object((0, 0, 0), exists=False),
        }
        assert predict_affected_objects(state, (0, 0, 0), (1, 0, 0), 0.05) == frozenset()

    def test_zero_length_motion_uses_spherical_clearance(self):
        state = {"glass": _object((0.02, 0.0, 0.0))}
        assert predict_affected_objects(state, (0, 0, 0), (0, 0, 0), 0.03) == frozenset({"glass"})

    def test_intended_target_can_be_excluded(self):
        state = {"red_mug": _object((1, 0, 0)), "glass": _object((0.5, 0.01, 0))}
        effects = predict_affected_objects(
            state, (0, 0, 0), (1, 0, 0), 0.05,
            exclude_objects=frozenset({"red_mug"}),
        )
        assert effects == frozenset({"glass"})

    def test_predicted_corridor_effect_blocks_protected_object(self):
        state = {
            "red_mug": _object((1, 0, 0)),
            "glass": _object((0.5, 0.01, 0)),
            "blue_bowl": _object((0, 1, 0)),
            "medicine_bottle": _object((0, -1, 0)),
        }
        effects = predict_affected_objects(
            state, (0, 0, 0), (1, 0, 0), 0.05,
            exclude_objects=frozenset({"red_mug"}),
        )
        allowed, reason = validate_action(
            "red_mug", canonical_example(), state=state, affected_objects=effects,
        )
        assert allowed is False
        assert "dont_move_glass" in reason

    @pytest.mark.parametrize("start,end,radius", [((0, 0), (1, 0, 0), 0.1), ((0, 0, 0), (1, 0), 0.1), ((0, 0, 0), (1, 0, 0), -0.1)])
    def test_rejects_invalid_geometry(self, start, end, radius):
        with pytest.raises(ValueError):
            predict_affected_objects({}, start, end, radius)


class TestWaypointPathEffects:
    def test_detects_object_near_second_leg_of_bent_path(self):
        state = {"glass": _object((1.0, 0.5, 0.02))}
        effects = predict_affected_objects_along_path(
            state,
            ((0, 0, 0), (1, 0, 0), (1, 1, 0)),
            0.05,
        )
        assert effects == frozenset({"glass"})

    def test_bent_path_does_not_flag_object_only_near_direct_chord(self):
        state = {"glass": _object((0.5, 0.5, 0.0))}
        effects = predict_affected_objects_along_path(
            state,
            ((0, 0, 0), (1, 0, 0), (1, 1, 0)),
            0.1,
        )
        assert effects == frozenset()

    def test_any_unsafe_leg_blocks_the_whole_action(self):
        state = {
            "red_mug": _object((1, 1, 0)),
            "glass": _object((1, 0.5, 0.01)),
            "blue_bowl": _object((2, 2, 0)),
            "medicine_bottle": _object((-1, -1, 0)),
        }
        effects = predict_affected_objects_along_path(
            state,
            ((0, 0, 0), (1, 0, 0), (1, 1, 0)),
            0.05,
            exclude_objects=frozenset({"red_mug"}),
        )
        allowed, _ = validate_action(
            "red_mug", canonical_example(), state=state, affected_objects=effects,
        )
        assert allowed is False

    @pytest.mark.parametrize("waypoints", [(), ((0, 0, 0),), ((0, 0, 0), (1, 0))])
    def test_rejects_incomplete_paths(self, waypoints):
        with pytest.raises(ValueError):
            predict_affected_objects_along_path({}, waypoints, 0.1)


class TestObjectExtents:
    def test_large_object_is_flagged_when_center_is_outside_robot_clearance(self):
        state = {"glass": _object((0.5, 0.14, 0.0))}
        point_effects = predict_affected_objects(
            state, (0, 0, 0), (1, 0, 0), clearance_radius=0.05,
        )
        extent_effects = predict_affected_objects(
            state, (0, 0, 0), (1, 0, 0), clearance_radius=0.05,
            object_radii={"glass": 0.10},
        )
        assert point_effects == frozenset()
        assert extent_effects == frozenset({"glass"})

    def test_unspecified_object_keeps_point_behavior(self):
        state = {"glass": _object((0.5, 0.14, 0.0))}
        assert predict_affected_objects(
            state, (0, 0, 0), (1, 0, 0), 0.05,
            object_radii={"another_object": 1.0},
        ) == frozenset()

    def test_zero_object_radius_matches_default(self):
        state = {"glass": _object((0.5, 0.04, 0.0))}
        assert predict_affected_objects(
            state, (0, 0, 0), (1, 0, 0), 0.05,
        ) == predict_affected_objects(
            state, (0, 0, 0), (1, 0, 0), 0.05,
            object_radii={"glass": 0.0},
        )

    def test_rejects_negative_object_radius(self):
        with pytest.raises(ValueError, match="object radii"):
            predict_affected_objects(
                {}, (0, 0, 0), (1, 0, 0), 0.05,
                object_radii={"glass": -0.1},
            )


class TestNavigationPlanAdapter:
    def test_real_2d_waypoint_shape_is_screened_by_intent_guard(self):
        state = {
            "red_mug": _object((1, 1, 0.5)),
            "glass": _object((1, 0.5, 0.5)),
            "blue_bowl": _object((2, 2, 0.5)),
            "medicine_bottle": _object((-1, -1, 0.5)),
        }
        allowed, reason, effects = screen_navigation_path(
            [(0, 0), (1, 0), (1, 1)],
            "red_mug",
            canonical_example(),
            state,
            travel_height=0.5,
            clearance_radius=0.05,
        )
        assert effects == frozenset({"glass"})
        assert allowed is False
        assert "dont_move_glass" in reason

    def test_safe_route_to_same_target_is_allowed(self):
        state = {
            "red_mug": _object((1, 1, 0.5)),
            "glass": _object((1, 0.5, 0.5)),
            "blue_bowl": _object((2, 2, 0.5)),
            "medicine_bottle": _object((-1, -1, 0.5)),
        }
        allowed, _, effects = screen_navigation_path(
            [(0, 0), (0, 1), (1, 1)],
            "red_mug",
            canonical_example(),
            state,
            travel_height=0.5,
            clearance_radius=0.05,
        )
        assert effects == frozenset()
        assert allowed is True

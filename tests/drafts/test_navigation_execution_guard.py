"""D-091: the Fetch executor honors navigation-path safety screening."""

from types import SimpleNamespace

import numpy as np

from atr.envs import tidy_up_replicacad_policies as policies
from atr.envs.navigation import NavigationOutcome, plan_path_avoiding_objects
from atr.feasibility.oracle import ObjectState
from atr.language.goal_graph import Goal, canonical_example


def _env():
    state = {
        "red_mug": ObjectState(True, np.array([1.0, 1.0, 0.5])),
        "glass": ObjectState(True, np.array([1.0, 0.5, 0.5])),
        "blue_bowl": ObjectState(True, np.array([2.0, 2.0, 0.5])),
        "medicine_bottle": ObjectState(True, np.array([-1.0, -1.0, 0.5])),
    }
    base_pose = SimpleNamespace(p=np.array([0.0, 0.0, 0.0]))
    unwrapped = SimpleNamespace(
        agent=SimpleNamespace(base_link=SimpleNamespace(pose=SimpleNamespace(sp=base_pose))),
        goal_graph=canonical_example(),
        _world_state=lambda: state,
    )
    return SimpleNamespace(unwrapped=unwrapped)


def test_constrained_planner_routes_around_an_affected_object():
    xs = np.arange(0.0, 2.1, 0.5)
    ys = np.arange(0.0, 2.1, 0.5)
    occupied = np.zeros((len(xs), len(ys)), dtype=bool)
    state = {"glass": ObjectState(True, np.array([1.0, 1.0, 0.5]))}

    path = plan_path_avoiding_objects(
        xs,
        ys,
        occupied,
        (0.0, 0.0),
        (2.0, 2.0),
        state,
        frozenset({"glass"}),
        clearance_radius=0.51,
    )

    assert path is not None
    assert all(np.linalg.norm(np.asarray(point) - np.array([1.0, 1.0])) > 0.51 for point in path)


def test_executor_stops_before_driving_a_route_that_threatens_a_constraint(monkeypatch):
    env = _env()
    monkeypatch.setattr(policies, "_get_or_build_grid", lambda _env: (None, None, None))
    monkeypatch.setattr(
        policies, "plan_path", lambda *_args: [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)],
    )
    monkeypatch.setattr(policies, "plan_path_avoiding_objects", lambda *_args, **_kwargs: None)
    drove = []
    monkeypatch.setattr(policies, "_drive_toward", lambda *_args: drove.append(True))

    outcome = policies._navigate_to(
        env, np.array([1.0, 1.0]), steps=100, target_object="red_mug",
    )

    assert outcome.steps_used == 0
    assert outcome.safety_screened is True
    assert "dont_move_glass" in outcome.blocked_reason
    assert outcome.replanned is False
    assert outcome.predicted_affected_objects == frozenset({"glass"})
    assert drove == []


def test_executor_does_not_drive_directly_when_grid_target_is_unreachable(monkeypatch):
    env = _env()
    monkeypatch.setattr(policies, "_get_or_build_grid", lambda _env: (None, None, None))
    monkeypatch.setattr(policies, "plan_path", lambda *_args: None)
    drove = []
    monkeypatch.setattr(policies, "_drive_toward", lambda *_args: drove.append(True))

    outcome = policies._navigate_to(
        env, np.array([1.0, 1.0]), steps=100, target_object="red_mug",
    )

    assert outcome.steps_used == 0
    assert outcome.failure_reason == "unreachable: no collision-free grid path"
    assert outcome.blocked_reason is None
    assert outcome.reached_target is False
    assert drove == []


def test_explicit_unguarded_ablation_preserves_direct_drive(monkeypatch):
    env = _env()
    monkeypatch.setattr(policies, "_get_or_build_grid", lambda _env: (None, None, None))
    monkeypatch.setattr(policies, "plan_path", lambda *_args: None)
    monkeypatch.setattr(policies, "_drive_toward", lambda *_args: 3)

    outcome = policies._navigate_to(
        env,
        np.array([1.0, 1.0]),
        steps=100,
        target_object="red_mug",
        enable_safety_screening=False,
    )

    assert outcome.steps_used == 3
    assert outcome.failure_reason is None
    assert outcome.safety_screened is False


def test_executor_replans_and_drives_a_safe_detour(monkeypatch):
    env = _env()
    monkeypatch.setattr(policies, "_get_or_build_grid", lambda _env: (None, None, None))
    monkeypatch.setattr(
        policies, "plan_path", lambda *_args: [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)],
    )
    avoided = []

    def safe_detour(*_args, **_kwargs):
        avoided.append(True)
        return [(0.0, 0.0), (0.0, 1.0), (1.0, 1.0)]

    monkeypatch.setattr(policies, "plan_path_avoiding_objects", safe_detour)
    monkeypatch.setattr(policies, "_drive_toward", lambda *_args: 3)

    outcome = policies._navigate_to(
        env, np.array([1.0, 1.0]), steps=100, target_object="red_mug",
    )

    assert avoided == [True]
    assert outcome.steps_used == 6
    assert outcome.safety_screened is True
    assert outcome.blocked_reason is None
    assert outcome.replanned is True
    assert outcome.predicted_affected_objects == frozenset({"glass"})


def test_executor_drives_a_screened_safe_route(monkeypatch):
    env = _env()
    monkeypatch.setattr(policies, "_get_or_build_grid", lambda _env: (None, None, None))
    monkeypatch.setattr(
        policies, "plan_path", lambda *_args: [(0.0, 0.0), (0.0, 1.0), (1.0, 1.0)],
    )
    monkeypatch.setattr(policies, "_drive_toward", lambda *_args: 3)

    outcome = policies._navigate_to(
        env, np.array([1.0, 1.0]), steps=100, target_object="red_mug",
    )

    assert outcome.steps_used == 6
    assert outcome.safety_screened is True
    assert outcome.blocked_reason is None
    assert outcome.replanned is False
    assert outcome.predicted_affected_objects == frozenset()


def test_attempt_result_exposes_navigation_adaptation_metadata(monkeypatch):
    unwrapped = SimpleNamespace(
        _exists={"potted_meat_can": False},
        _elapsed_control_steps=7,
    )
    env = SimpleNamespace(unwrapped=unwrapped)
    monkeypatch.setattr(
        policies,
        "_navigate_to",
        lambda *_args, **_kwargs: NavigationOutcome(
            steps_used=4,
            replanned=True,
            predicted_affected_objects=frozenset({"master_chef_can"}),
        ),
    )

    result = policies.attempt_goal(
        env,
        Goal(id="place_can", predicate="on_tray", target_object="potted_meat_can"),
        np.zeros(3),
    )

    assert result["navigation_replanned"] is True
    assert result["navigation_safety_screened"] is True
    assert result["predicted_affected_objects"] == ["master_chef_can"]

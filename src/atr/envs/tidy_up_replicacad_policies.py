"""Static / feasibility-aware / naive-substitution policies for the
ReplicaCAD + Fetch version of TidyUp (tidy_up_env_replicacad.py).

Promoted to src/atr/ 2026-08-04 (D-048), alongside tidy_up_env_replicacad.py
and navigation.py -- see ai-notes/decisions.md. `_TRAY_POSITION`/
`_TRAY_HALF_SIZES` were already imported from tidy_up_env_replicacad.py
rather than duplicated (unlike D-046's canonical-env bug), so no fix
needed there; `_LAST_KNOWN_POSITIONS` are standalone empirical fallback
positions with no `_OBJECT_SPECS`-equivalent source of truth to derive
from in this env (real YCB objects, not hand-placed boxes), so those stay
as calibrated literals too, same as clip_feasibility.py's
`_OBJECT_VISUAL_CONFIG`.

Same goal_graph, oracle_feasibility, and intent_guard logic as the panda and
humanoid variants. What's genuinely new here: "attempt a goal" now means
*navigate* to the object (it may be a room away) before reaching for it —
this scene doesn't put everything within one fixed arm's reach. A naive
point-and-drive controller physically got stuck against a real wall/doorway
(confirmed via raycast, not assumed — see navigation.py's module docstring),
so this plans a path with a grid + Dijkstra planner first, then follows the
waypoints with the same proportional go-to-pose controller. Navigation cost
is real (measured in actual simulation steps), which makes the wasted-effort
gap between static and feasibility-aware policies larger and more realistic
than the fixed-base versions: skipping an infeasible goal here saves a real
cross-apartment trip, not just an arm swing.
"""

from __future__ import annotations

import numpy as np

from atr.language.goal_graph import Goal, GoalGraph
from atr.feasibility.oracle import goal_achieved
from atr.policies import baselines
from atr.envs.navigation import (
    NavigationOutcome,
    build_occupancy_grid,
    plan_path,
    plan_path_avoiding_objects,
    screen_navigation_path,
)
from atr.envs.tidy_up_env_replicacad import _TRAY_HALF_SIZES, _TRAY_POSITION

# Covers spawn (-1, 0) plus every goal/constraint object position used in
# this scenario, with margin -- see tidy_up_env_replicacad.py's alias map.
_GRID_X_RANGE = (-2.5, 1.0)
_GRID_Y_RANGE = (-1.5, 1.5)

# Last-known positions for objects that may no longer exist when a policy
# tries to reach them (same role as manipulation_skill_spike's fallback) --
# from the real ReplicaCADSetTableTrain seed=0 build/init config (see
# tidy_up_env_replicacad.py's module docstring).
_LAST_KNOWN_POSITIONS = {
    "potted_meat_can": np.array([0.29, 0.09, 0.68]),
    "bowl": np.array([-2.13, -0.83, 0.53]),
    "master_chef_can": np.array([0.81, 0.37, 0.71]),
    "cracker_box": np.array([3.72, -0.62, 0.52]),
}


def _yaw(pose) -> float:
    q = pose.q  # sapien.Pose quaternion is (w, x, y, z)
    return float(2 * np.arctan2(q[3], q[0]))


def _get_or_build_grid(env):
    """Occupancy grid is built once per env instance and reused across every
    attempt_goal() call in the episode — the static apartment architecture
    doesn't change (only a couple of small objects are added/removed by
    interventions, which don't meaningfully affect navigability)."""
    if getattr(env.unwrapped, "_nav_grid", None) is None:
        px = env.unwrapped.scene.px
        # 0.3 (a literal Fetch-base-radius-sized margin) left every doorway
        # in this scene fully sealed in the discretized grid -- no path
        # existed at all. 0.2 is the largest margin that still finds a path;
        # verified empirically (see navigation.py), not just assumed safe.
        env.unwrapped._nav_grid = build_occupancy_grid(
            px, _GRID_X_RANGE, _GRID_Y_RANGE, robot_radius=0.2
        )
    return env.unwrapped._nav_grid


def _drive_toward(env, target_xy: np.ndarray, steps: int, distance_tol: float):
    """One proportional go-to-pose step toward a single (possibly
    intermediate) waypoint. Fetch's base sub-controller is
    PDBaseForwardVelControllerConfig — action = [forward_velocity,
    turn_velocity], both in [-1, 1] (calibrated empirically, see
    tidy_up_env_replicacad.py's module docstring). Returns steps actually used."""
    used = 0
    for _ in range(steps):
        base_pose = env.unwrapped.agent.base_link.pose.sp
        pos = base_pose.p[:2]
        yaw = _yaw(base_pose)
        to_target = np.asarray(target_xy[:2]) - pos
        dist = float(np.linalg.norm(to_target))
        if dist < distance_tol:
            break
        desired_yaw = float(np.arctan2(to_target[1], to_target[0]))
        yaw_err = (desired_yaw - yaw + np.pi) % (2 * np.pi) - np.pi
        action = np.zeros(9, dtype=np.float32)
        action[8] = np.clip(yaw_err / 0.3, -1, 1)  # turn
        action[7] = np.clip(dist, 0.0, 1.0) if abs(yaw_err) < 0.5 else 0.0  # forward, only once facing target
        env.step(action)
        used += 1
    return used


def _navigate_to(
    env,
    target_xy: np.ndarray,
    steps: int,
    target_object: str,
    distance_tol: float = 0.5,
    allow_replan: bool = True,
) -> NavigationOutcome:
    """Plans a path around real obstacles (see navigation.py), then drives
    through each waypoint in sequence. Before executing, screens that exact
    path through the intent guard; a route predicted to disturb a protected
    object is stopped with zero motion and its reason returned to the caller.
    Falls back to a screened direct path if no grid path exists. Returns
    a structured outcome so evaluation can distinguish ordinary execution,
    successful safety replanning, and a fail-closed stop. `allow_replan=False`
    is D-097's explicit stop-only ablation; production behavior defaults True."""
    xs, ys, occupied = _get_or_build_grid(env)
    start_xy = env.unwrapped.agent.base_link.pose.sp.p[:2]
    waypoints = plan_path(xs, ys, occupied, start_xy, target_xy[:2])
    if waypoints is None:
        waypoints = [tuple(start_xy), tuple(target_xy[:2])]

    state = env.unwrapped._world_state()
    allowed, reason, effects = screen_navigation_path(
        waypoints,
        target_object,
        env.unwrapped.goal_graph,
        state,
        travel_height=0.5,
        clearance_radius=0.2,
    )
    if not allowed:
        initial_effects = effects
        if not allow_replan:
            return NavigationOutcome(
                steps_used=0,
                blocked_reason=reason,
                predicted_affected_objects=initial_effects,
            )
        alternate = plan_path_avoiding_objects(
            xs,
            ys,
            occupied,
            start_xy,
            target_xy[:2],
            state,
            effects,
            clearance_radius=0.2,
        )
        if alternate is None:
            return NavigationOutcome(
                steps_used=0,
                blocked_reason=reason,
                predicted_affected_objects=initial_effects,
            )
        alternate_allowed, alternate_reason, _ = screen_navigation_path(
            alternate,
            target_object,
            env.unwrapped.goal_graph,
            state,
            travel_height=0.5,
            clearance_radius=0.2,
        )
        if not alternate_allowed:
            return NavigationOutcome(
                steps_used=0,
                blocked_reason=alternate_reason,
                replanned=True,
                predicted_affected_objects=initial_effects,
            )
        waypoints = alternate
        replanned = True
    else:
        initial_effects = effects
        replanned = False

    used = 0
    remaining = steps
    for i, waypoint in enumerate(waypoints[1:]):  # [0] is the start cell itself
        is_last = i == len(waypoints) - 2
        tol = distance_tol if is_last else max(distance_tol, 0.35)
        step_budget = max(1, remaining // max(1, len(waypoints) - 1 - i))
        used_here = _drive_toward(env, np.array(waypoint), step_budget, tol)
        used += used_here
        remaining -= used_here
        if remaining <= 0:
            break
    return NavigationOutcome(
        steps_used=used,
        replanned=replanned,
        predicted_affected_objects=initial_effects,
    )


def attempt_goal(
    env,
    goal: Goal,
    tray_slot_xyz: np.ndarray,
    nav_steps: int = 250,
    allow_replan: bool = True,
) -> dict:
    exists = env.unwrapped._exists[goal.target_object]
    target_xy = (
        env.unwrapped._get_actor(goal.target_object).pose.sp.p
        if exists
        else _LAST_KNOWN_POSITIONS[goal.target_object]
    )

    before = env.unwrapped._elapsed_control_steps
    navigation = _navigate_to(
        env,
        target_xy,
        steps=nav_steps,
        target_object=goal.target_object,
        allow_replan=allow_replan,
    )
    steps_used = env.unwrapped._elapsed_control_steps - before

    navigation_metadata = {
        "navigation_safety_screened": navigation.safety_screened,
        "navigation_replanned": navigation.replanned,
        "predicted_affected_objects": sorted(navigation.predicted_affected_objects),
    }
    if navigation.blocked_reason is not None:
        return {
            "achieved": False,
            "steps_used": steps_used,
            "skipped": True,
            "blocked_reason": navigation.blocked_reason,
            **navigation_metadata,
        }

    if not exists:
        return {
            "achieved": False, "steps_used": steps_used, "skipped": False,
            **navigation_metadata,
        }

    import sapien

    obj = env.unwrapped._get_actor(goal.target_object)
    obj.set_pose(sapien.Pose(p=tray_slot_xyz))
    state = env.unwrapped._world_state()
    achieved = goal_achieved(goal, state, _TRAY_POSITION, _TRAY_HALF_SIZES)
    return {
        "achieved": achieved, "steps_used": steps_used, "skipped": False,
        **navigation_metadata,
    }


# Re-exported for existing callers that import this privately -- see
# atr.policies.baselines for the real implementation.
_summarize = baselines._summarize


_TRAY_SLOTS = [
    _TRAY_POSITION + np.array([0.1, 0.0, 0.0]),
    _TRAY_POSITION + np.array([-0.1, 0.0, 0.0]),
]


def static_policy(env, graph: GoalGraph = None) -> dict:
    from atr.envs.tidy_up_env_replicacad import replicacad_example

    return baselines.static_policy(env, graph or replicacad_example(), attempt_goal, _TRAY_SLOTS)


def feasibility_aware_policy(env, graph: GoalGraph = None) -> dict:
    from atr.envs.tidy_up_env_replicacad import replicacad_example

    return baselines.feasibility_aware_policy(env, graph or replicacad_example(), attempt_goal, _TRAY_SLOTS)


def naive_substitution_policy(env, graph: GoalGraph = None, use_intent_guard: bool = False) -> dict:
    from atr.envs.tidy_up_env_replicacad import replicacad_example

    return baselines.naive_substitution_policy(
        env, graph or replicacad_example(), attempt_goal, _TRAY_SLOTS, use_intent_guard=use_intent_guard,
    )

"""Static / feasibility-aware / naive-substitution policies for the
Unitree G1 humanoid version of TidyUp (tidy_up_env_humanoid.py).

Same policy logic and metrics as policy_baselines.py (panda arm) — same
goal_graph, oracle_feasibility, and intent_guard modules, genuinely
embodiment-agnostic as designed. The only thing that changes is how
"attempt a goal" is realized physically: no Cartesian controller exists for
this robot (see tidy_up_env_humanoid.py's module docstring), so `_reach`
drives pre-calibrated right-arm joint targets instead of proportional IK
toward an xyz target. Placement still uses the same teleport-on-success
abstraction as the panda version, for the same reason.
"""

from __future__ import annotations

import numpy as np
import sapien

from atr.language.goal_graph import Goal, GoalGraph, canonical_example
from atr.constraints.intent_guard import validate_action
from atr.feasibility.oracle import constraint_violated, goal_achieved, goal_feasible
from task_schema_draft.tidy_up_env_humanoid import _NEUTRAL_QPOS, _REACH_CONFIGS

_TRAY_POSITION = np.array([0.0, -0.13, 0.698])
_TRAY_HALF_SIZES = (0.12, 0.15, 0.02)
_TRAY_SLOTS = [
    _TRAY_POSITION + np.array([0.05, 0.0, 0.0]),
    _TRAY_POSITION + np.array([-0.05, 0.0, 0.0]),
]
_DEFAULT_REACH = _REACH_CONFIGS["red_mug"]


def _reach(env, joint_targets: dict, steps: int):
    body_joints = env.unwrapped.agent.body_joints
    idx = {name: i for i, name in enumerate(body_joints)}
    qpos = _NEUTRAL_QPOS.copy()
    for name, value in joint_targets.items():
        qpos[idx[name]] = value
    for _ in range(steps):
        env.step(qpos)


def attempt_goal(env, goal: Goal, tray_slot_xyz: np.ndarray, reach_steps: int = 25) -> dict:
    exists = env.unwrapped._exists[goal.target_object]
    reach_config = _REACH_CONFIGS.get(goal.target_object, _DEFAULT_REACH)

    before = env.unwrapped._elapsed_control_steps
    _reach(env, reach_config, reach_steps)
    steps_used = env.unwrapped._elapsed_control_steps - before

    if not exists:
        return {"achieved": False, "steps_used": steps_used, "skipped": False}

    obj = env.unwrapped._objects[goal.target_object]
    obj.set_pose(sapien.Pose(p=tray_slot_xyz))
    state = env.unwrapped._world_state()
    achieved = goal_achieved(goal, state, _TRAY_POSITION, _TRAY_HALF_SIZES)
    return {"achieved": achieved, "steps_used": steps_used, "skipped": False}


def _summarize(per_goal: dict) -> dict:
    return {
        "per_goal": per_goal,
        "goals_achieved": sum(r["achieved"] for r in per_goal.values()),
        "total_steps": sum(r["steps_used"] for r in per_goal.values()),
        "wasted_steps": sum(
            r["steps_used"] for r in per_goal.values() if not r["achieved"] and not r["skipped"]
        ),
    }


def static_policy(env, graph: GoalGraph = None) -> dict:
    graph = graph or canonical_example()
    per_goal = {
        goal.id: attempt_goal(env, goal, _TRAY_SLOTS[i]) for i, goal in enumerate(graph.goals)
    }
    return _summarize(per_goal)


def feasibility_aware_policy(env, graph: GoalGraph = None) -> dict:
    graph = graph or canonical_example()
    per_goal = {}
    for i, goal in enumerate(graph.goals):
        state = env.unwrapped._world_state()
        if not goal_feasible(goal, state):
            per_goal[goal.id] = {"achieved": False, "steps_used": 0, "skipped": True}
            continue
        per_goal[goal.id] = attempt_goal(env, goal, _TRAY_SLOTS[i])
    return _summarize(per_goal)


def naive_substitution_policy(env, graph: GoalGraph = None, use_intent_guard: bool = False) -> dict:
    graph = graph or canonical_example()
    # Let physics settle before establishing the never-move/upright baseline
    # -- objects are spawned slightly above the counter's real surface (see
    # tidy_up_env_humanoid.py) and drop a small amount in the first few
    # steps. Capturing initial_state before that settling finishes would
    # register the settle itself as a false "moved" violation. Bypasses
    # env.unwrapped._world_state() directly, not evaluate()'s own settle-
    # window fix, so it needs this explicitly.
    for _ in range(5):
        env.step(_NEUTRAL_QPOS)
    initial_state = env.unwrapped._world_state()
    per_goal = {}
    substitution_attempted = False

    for i, goal in enumerate(graph.goals):
        state = env.unwrapped._world_state()
        if goal_feasible(goal, state):
            per_goal[goal.id] = attempt_goal(env, goal, _TRAY_SLOTS[i])
            continue

        if use_intent_guard:
            allowed, reason = validate_action("glass", graph)
        else:
            allowed, reason = True, "unchecked (no intent guard)"

        if not allowed:
            per_goal[goal.id] = {
                "achieved": False, "steps_used": 0, "skipped": True,
                "substitution_attempted": False, "blocked_reason": reason,
            }
            continue

        substitution_attempted = True
        fake_goal = Goal(id=f"substitute_for_{goal.id}", predicate="on_tray", target_object="glass")
        substitution_result = attempt_goal(env, fake_goal, _TRAY_SLOTS[i])
        per_goal[goal.id] = {
            "achieved": False,
            "steps_used": substitution_result["steps_used"],
            "skipped": False,
            "substitution_attempted": True,
        }

    final_state = env.unwrapped._world_state()
    glass_constraint = next(c for c in graph.constraints if c.target_object == "glass")
    result = _summarize(per_goal)
    result["dont_move_glass_violated"] = constraint_violated(glass_constraint, initial_state, final_state)
    result["substitution_attempted"] = substitution_attempted
    return result

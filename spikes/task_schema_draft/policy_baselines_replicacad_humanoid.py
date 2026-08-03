"""Static / feasibility-aware / naive-substitution policies for the
fixed-placement G1-in-ReplicaCAD version of TidyUp
(tidy_up_env_replicacad_humanoid.py).

No navigation here — G1 is fixed-base, placed once. "Attempt a goal" means
a joint-space arm reach (same approach as policy_baselines_humanoid.py's
kitchen-counter variant), then teleport-on-success, same abstraction as
every other variant.
"""

from __future__ import annotations

import numpy as np
import sapien

from atr.language.goal_graph import Goal, GoalGraph
from atr.constraints.intent_guard import validate_action
from atr.feasibility.oracle import constraint_violated, goal_achieved, goal_feasible
from task_schema_draft.tidy_up_env_replicacad_humanoid import (
    _LAST_KNOWN_POSITIONS,
    _NEUTRAL_QPOS,
    _REACH_CONFIGS,
    _TRAY_HALF_SIZES,
    _TRAY_POSITION,
    replicacad_humanoid_example,
)

_TRAY_SLOTS = [
    _TRAY_POSITION + np.array([0.08, 0.0, 0.0]),
    _TRAY_POSITION + np.array([-0.08, 0.0, 0.0]),
]
_DEFAULT_REACH = _REACH_CONFIGS["potted_meat_can"]


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

    obj = env.unwrapped._get_actor(goal.target_object)
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
    graph = graph or replicacad_humanoid_example()
    per_goal = {
        goal.id: attempt_goal(env, goal, _TRAY_SLOTS[i]) for i, goal in enumerate(graph.goals)
    }
    return _summarize(per_goal)


def feasibility_aware_policy(env, graph: GoalGraph = None) -> dict:
    graph = graph or replicacad_humanoid_example()
    per_goal = {}
    for i, goal in enumerate(graph.goals):
        state = env.unwrapped._world_state()
        if not goal_feasible(goal, state):
            per_goal[goal.id] = {"achieved": False, "steps_used": 0, "skipped": True}
            continue
        per_goal[goal.id] = attempt_goal(env, goal, _TRAY_SLOTS[i])
    return _summarize(per_goal)


def naive_substitution_policy(env, graph: GoalGraph = None, use_intent_guard: bool = False) -> dict:
    graph = graph or replicacad_humanoid_example()
    # Same fix as policy_baselines_replicacad.py / _humanoid.py: this reads
    # state directly via _world_state(), bypassing evaluate()'s own
    # settle-window fix, so real objects still settling onto real surfaces
    # in the first few steps would otherwise register as a false
    # never-move violation before anything has touched them.
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
            allowed, reason = validate_action("bowl", graph)
        else:
            allowed, reason = True, "unchecked (no intent guard)"

        if not allowed:
            per_goal[goal.id] = {
                "achieved": False, "steps_used": 0, "skipped": True,
                "substitution_attempted": False, "blocked_reason": reason,
            }
            continue

        substitution_attempted = True
        fake_goal = Goal(id=f"substitute_for_{goal.id}", predicate="on_tray", target_object="bowl")
        substitution_result = attempt_goal(env, fake_goal, _TRAY_SLOTS[i])
        per_goal[goal.id] = {
            "achieved": False,
            "steps_used": substitution_result["steps_used"],
            "skipped": False,
            "substitution_attempted": True,
        }

    final_state = env.unwrapped._world_state()
    guarded_constraint = next(c for c in graph.constraints if c.target_object == "bowl")
    result = _summarize(per_goal)
    result["dont_move_bowl_violated"] = constraint_violated(guarded_constraint, initial_state, final_state)
    result["substitution_attempted"] = substitution_attempted
    return result

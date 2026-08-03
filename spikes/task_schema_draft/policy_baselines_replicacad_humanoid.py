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
from atr.feasibility.oracle import goal_achieved
from atr.policies import baselines
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


# Re-exported for existing callers that import this privately -- see
# atr.policies.baselines for the real implementation.
_summarize = baselines._summarize


def static_policy(env, graph: GoalGraph = None) -> dict:
    return baselines.static_policy(env, graph or replicacad_humanoid_example(), attempt_goal, _TRAY_SLOTS)


def feasibility_aware_policy(env, graph: GoalGraph = None) -> dict:
    return baselines.feasibility_aware_policy(
        env, graph or replicacad_humanoid_example(), attempt_goal, _TRAY_SLOTS,
    )


def naive_substitution_policy(env, graph: GoalGraph = None, use_intent_guard: bool = False) -> dict:
    # settle_steps=5: same fix as policy_baselines_humanoid.py -- this
    # reads state directly via _world_state(), bypassing evaluate()'s own
    # settle-window fix, so real objects still settling onto real surfaces
    # in the first few steps would otherwise register as a false
    # never-move violation before anything has touched them.
    return baselines.naive_substitution_policy(
        env, graph or replicacad_humanoid_example(), attempt_goal, _TRAY_SLOTS,
        use_intent_guard=use_intent_guard, settle_steps=5, settle_action=_NEUTRAL_QPOS,
    )

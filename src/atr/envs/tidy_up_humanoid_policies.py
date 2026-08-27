"""Static / feasibility-aware / naive-substitution policies for the
Unitree G1 humanoid version of TidyUp (tidy_up_env_humanoid.py).

Promoted to src/atr/ 2026-08-04 (D-047), alongside tidy_up_env_humanoid.py
-- see ai-notes/decisions.md. `_TRAY_POSITION`'s z (0.698) is deliberately
*not* derived from `_OBJECT_SPECS["tray"]`'s spawn z (0.755) the way
D-046 derived the canonical env's tray position -- checked first, not
assumed the same fix applied: `tidy_up_env_humanoid.py`'s own `evaluate()`
docstring explains objects are spawned at an assumed counter height that
doesn't match the counter's real collision surface and settle a small
amount in the first few steps, so 0.698 is very likely the real,
empirically-observed resting height, not a stale duplicate of 0.755 the
spawn height. Left exactly as-is.

Same policy logic and metrics as atr.envs.tidy_up_policies (panda arm,
D-046) — same goal_graph, oracle_feasibility, and intent_guard modules,
genuinely embodiment-agnostic as designed. The only thing that changes is
how "attempt a goal" is realized physically: no Cartesian controller
exists for this robot (see tidy_up_env_humanoid.py's module docstring),
so `_reach` drives pre-calibrated right-arm joint targets instead of
proportional IK toward an xyz target. Placement still uses the same
teleport-on-success abstraction as the panda version, for the same
reason.
"""

from __future__ import annotations

import numpy as np
import sapien

from atr.language.goal_graph import Goal, GoalGraph, canonical_example
from atr.feasibility.oracle import goal_achieved
from atr.policies import baselines
from atr.envs.tidy_up_env_humanoid import _NEUTRAL_QPOS, _REACH_CONFIGS

_TRAY_POSITION = np.array([0.0, -0.13, 0.698])
_TRAY_HALF_SIZES = (0.12, 0.15, 0.02)
_TRAY_SLOTS = [
    _TRAY_POSITION + np.array([0.05, 0.0, 0.0]),
    _TRAY_POSITION + np.array([-0.05, 0.0, 0.0]),
]
_DEFAULT_REACH = _REACH_CONFIGS["red_mug"]

# Effect model for the two fixed, hand-calibrated semantic reach skills.
# The blue-bowl configuration was measured to sweep the protected glass off
# the counter (about 1.39 m displacement); the red-mug configuration leaves
# it stationary.  This is deliberately skill-level and scoped to these fixed
# controllers, not presented as a general robot-link collision predictor.
# It lets the intent guard screen real execution effects instead of guarding
# only the semantically-invalid substitution branch.
_REACH_PREDICTED_EFFECTS = {
    "red_mug": frozenset(),
    "blue_bowl": frozenset({"glass"}),
}


def _predict_goal_effects(env, goal: Goal) -> frozenset[str]:
    del env  # interface leaves room for a future state-conditioned skill model
    return _REACH_PREDICTED_EFFECTS.get(goal.target_object, frozenset())


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


# Re-exported for existing callers that import this privately -- see
# atr.policies.baselines for the real implementation.
_summarize = baselines._summarize


def static_policy(env, graph: GoalGraph = None) -> dict:
    return baselines.static_policy(env, graph or canonical_example(), attempt_goal, _TRAY_SLOTS)


def feasibility_aware_policy(env, graph: GoalGraph = None) -> dict:
    return baselines.feasibility_aware_policy(env, graph or canonical_example(), attempt_goal, _TRAY_SLOTS)


def naive_substitution_policy(env, graph: GoalGraph = None, use_intent_guard: bool = False) -> dict:
    # settle_steps=5: objects are spawned slightly above the counter's real
    # surface (see tidy_up_env_humanoid.py) and drop a small amount in the
    # first few steps -- capturing initial_state before that settling
    # finishes would register the settle itself as a false "moved"
    # violation (this bypasses evaluate()'s own settle-window fix by
    # reading _world_state() directly, so it needs this explicitly).
    return baselines.naive_substitution_policy(
        env, graph or canonical_example(), attempt_goal, _TRAY_SLOTS,
        use_intent_guard=use_intent_guard, settle_steps=5, settle_action=_NEUTRAL_QPOS,
        predict_goal_effects_fn=_predict_goal_effects,
    )

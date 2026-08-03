"""Static vs. feasibility-aware policy baselines — the first runnable test
of H2 from docs/01-problem-statement-and-motivation.md: "conditioning
strategy selection on per-goal feasibility estimates outperforms a static
language-conditioned policy after irreversible changes."

Both policies attempt `canonical_example()`'s goals in order (mug, then
bowl) using real arm motion for the "attempt" (reach) phase. They differ in
exactly one thing: whether they check `goal_feasible()` before attempting.

Scope note: a successful attempt places the object by teleporting it onto
the tray (`actor.set_pose`) rather than re-running a full IK grasp-lift-place
sequence. The grasp mechanic itself was already validated separately and
robustly in `maniskill_humanoid_spike/manipulation_skill_spike.py` (5/5 on
PickCube-v1) — re-risking that across two differently-positioned objects
isn't what this comparison is testing. What *is* under test — real arm
motion time spent reaching for a goal, feasible or not — is not abstracted:
the reach phase always runs to completion (or is skipped by the
feasibility-aware policy), so the "wasted time" metric below is genuine.
"""

from __future__ import annotations

import numpy as np
import sapien

from atr.language.goal_graph import Goal, GoalGraph, canonical_example
from atr.feasibility.oracle import goal_achieved
from atr.policies import baselines

_TRAY_POSITION = np.array([0.4, 0.0, 0.005])
_TRAY_HALF_SIZES = (0.15, 0.2, 0.005)
_TRAY_SLOTS = [
    _TRAY_POSITION + np.array([0.0, -0.08, 0.0]),
    _TRAY_POSITION + np.array([0.0, 0.08, 0.0]),
]

_LAST_KNOWN_POSITION = {
    "red_mug": np.array([0.15, -0.15, 0.04]),
    "blue_bowl": np.array([0.15, 0.15, 0.025]),
}


def _go_to(env, target_xyz: np.ndarray, gripper: float, steps: int, tol: float = 0.005):
    for _ in range(steps):
        tcp = env.unwrapped.agent.tcp.pose.sp.p
        delta = np.clip((target_xyz - tcp) / 0.1, -1, 1)
        action = np.array([delta[0], delta[1], delta[2], gripper], dtype=np.float32)
        env.step(action)
        if np.linalg.norm(target_xyz - env.unwrapped.agent.tcp.pose.sp.p) < tol:
            break


def attempt_goal(env, goal: Goal, tray_slot_xyz: np.ndarray, reach_steps: int = 25) -> dict:
    """Reach for the goal's target object; if it still exists, place it on
    the tray. Real arm motion either way — see module docstring."""
    exists = env.unwrapped._exists[goal.target_object]
    obj_xyz = (
        env.unwrapped._objects[goal.target_object].pose.sp.p.copy()
        if exists
        else _LAST_KNOWN_POSITION[goal.target_object]
    )

    before = env.unwrapped._elapsed_control_steps
    _go_to(env, obj_xyz + np.array([0.0, 0.0, 0.08]), gripper=1.0, steps=reach_steps)
    steps_used = env.unwrapped._elapsed_control_steps - before

    if not exists:
        return {"achieved": False, "steps_used": steps_used, "skipped": False}

    obj = env.unwrapped._objects[goal.target_object]
    obj.set_pose(sapien.Pose(p=tray_slot_xyz))
    state = env.unwrapped._world_state()
    achieved = goal_achieved(goal, state, _TRAY_POSITION, _TRAY_HALF_SIZES)
    return {"achieved": achieved, "steps_used": steps_used, "skipped": False}


# Re-exported for existing callers (rl_policy.py) that import this
# privately -- see atr.policies.baselines for the real implementation.
_summarize = baselines._summarize


def static_policy(env, graph: GoalGraph = None) -> dict:
    """Attempts every goal in order, regardless of feasibility."""
    return baselines.static_policy(env, graph or canonical_example(), attempt_goal, _TRAY_SLOTS)


def feasibility_aware_policy(env, graph: GoalGraph = None) -> dict:
    """Checks goal_feasible() (a privileged-state query, ~zero cost) and
    goal_dependencies_satisfied() (D-037) before committing to the
    physical attempt; skips a goal immediately if either fails. See
    atr.policies.baselines.feasibility_aware_policy() for the shared
    implementation (D-040)."""
    return baselines.feasibility_aware_policy(env, graph or canonical_example(), attempt_goal, _TRAY_SLOTS)


def naive_substitution_policy(env, graph: GoalGraph = None, use_intent_guard: bool = False) -> dict:
    """The "invalid agent" from docs/01's own worked example: rather than
    accepting an infeasible goal, it substitutes an unrequested object (the
    glass, canonical_example()'s only never_move-constrained object) to
    fill the tray slot instead — which never actually satisfies the real
    goal and violates the `dont_move_glass` hard constraint. With
    `use_intent_guard=True`, `validate_action()` is checked before the
    substitution and blocks it. See atr.policies.baselines.
    naive_substitution_policy() for the shared implementation (D-040)."""
    return baselines.naive_substitution_policy(
        env, graph or canonical_example(), attempt_goal, _TRAY_SLOTS, use_intent_guard=use_intent_guard,
    )

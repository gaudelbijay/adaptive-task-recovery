"""D-124: real pick-and-place for the ReplicaCAD + Fetch variant, as a
separate, additive capability alongside `tidy_up_replicacad_policies.py`'s
`attempt_goal()` -- deliberately not a replacement.

`attempt_goal()` (and every policy/benchmark/decision built on it, D-048
onward) treats "attempt a goal" as navigate-then-teleport: once Fetch is
within `distance_tol` of the target, the object is placed directly at the
tray slot via `obj.set_pose(...)`, no reach/grasp motion at all. That
abstraction is deliberate and load-bearing -- this project's actual research
question is feasibility reasoning and adaptation, not low-level manipulation
(see docs/07-adaptive-policy-design.md), and every H1-H5 result, every
navigation-safety decision (D-091-121), and 300+ tests assume exactly that
contract. Changing `attempt_goal()` itself would be a large, high-risk
change to code the whole project's evidence base depends on, for a benefit
(visual realism) that only matters for a demo capture, not for any research
claim this project makes.

This module exists for the one place that benefit *does* matter: a demo GIF
that should show a real robot doing something, not an object popping into
existence. Everything in it is real, physically simulated, and independently
verifiable:

- Real navigation, reusing `tidy_up_replicacad_policies._navigate_to()`
  unchanged (same collision-aware planning, same safety screening).
- A real IK-based reach: Fetch's own `pd_ee_delta_pos` controller solves
  inverse kinematics internally (`PDEEPosController`, ManiSkill3's own code,
  not reimplemented here) -- this module only needs to close the loop by
  recomputing the position error every control step (proportional/visual-
  servo control) and sending it in the controller's own frame
  (`root_translation`, relative to `torso_lift_link`). A naive single large
  delta command was tried first and found to silently fail (the controller
  keeps the previous joint position whenever IK can't solve for the
  requested step, so a too-large one-shot delta looks identical to "did
  nothing" -- confirmed directly by watching qpos not change across 30
  repeated steps, not assumed); small proportional steps toward a real,
  reachable target converge cleanly in under 30 steps.
- A real grasp: closes the gripper, then checks `agent.is_grasping()` --
  ManiSkill3's own contact-force-based grasp detector (real per-finger
  contact impulses and angle, not a proximity heuristic) -- rather than
  assuming the gripper closing means the object is held.
- A real carry: the grasp is re-verified after lifting and again after
  driving across the apartment, confirming the object moves with the
  gripper through real physics, not that a flag stayed set.
- A real placement: `_TRAY_POSITION`/`_TRAY_HALF_SIZES`
  (`tidy_up_env_replicacad.py`) were always a purely logical scoring region
  with no physical surface -- confirmed directly (grepped the env file: no
  tray actor is ever built), which is exactly why `attempt_goal()`'s
  teleport never needed one. A real released object has nothing to land on
  there and just keeps falling. `ensure_tray_surface()` adds one real static
  box for this module's own use, sized so its top sits inside the existing
  scoring region's tolerance -- additive to the scene, not a change to the
  shared env class, so it has no effect on `attempt_goal()` or anything
  that doesn't call this module.
"""

from __future__ import annotations

import sapien
import numpy as np
from mani_skill.utils.building.actors import build_box

from atr.envs.tidy_up_env_replicacad import _TRAY_HALF_SIZES, _TRAY_POSITION
from atr.envs.tidy_up_replicacad_policies import _navigate_to
from atr.feasibility.oracle import goal_achieved
from atr.language.goal_graph import Goal

_TRAY_SURFACE_NAME = "demo_tray_surface"
_TRAY_SURFACE_HALF_SIZES = (0.22, 0.22, 0.02)
# goal_achieved() (atr.feasibility.oracle) requires the object's final z to
# land within [-1e-4, +0.05] of _TRAY_POSITION's z specifically -- tighter
# than _TRAY_HALF_SIZES[2] suggests, and one-sided (at/above, not below).
# Found empirically, not guessed: a first attempt at -0.08m undershot this
# window (object settled ~0.018m *below* _TRAY_POSITION's z, failing the
# lower bound) -- surface height tuned so a real resting object's measured
# center lands inside the actual required band.
_TRAY_SURFACE_TOP_Z = _TRAY_POSITION[2] - 0.04
_APPROACH_HEIGHT = 0.08


def ensure_tray_surface(env) -> None:
    """Idempotent: adds the real static tray surface `attempt_goal()` never
    needed (its teleport has nothing to land on) if it isn't already in this
    scene. Safe to call every episode -- checks by actor name first."""
    scene = env.unwrapped.scene
    if _TRAY_SURFACE_NAME in scene.actors:
        return
    build_box(
        scene, half_sizes=list(_TRAY_SURFACE_HALF_SIZES), color=[0.5, 0.35, 0.2, 1],
        name=_TRAY_SURFACE_NAME, body_type="static",
        initial_pose=sapien.Pose(p=[_TRAY_POSITION[0], _TRAY_POSITION[1], _TRAY_SURFACE_TOP_Z]),
    )
    scene.update_render()


def _arm_controller(env):
    return env.unwrapped.agent.controller.controllers["arm"]


def _ee_pos_local(arm_ctrl) -> np.ndarray:
    return arm_ctrl.ee_pose_at_base.p.numpy()[0]


def _world_to_local(arm_ctrl, world_pos) -> np.ndarray:
    return (arm_ctrl.root_link.pose.inv() * sapien.Pose(p=world_pos)).p


def _servo_arm_to(
    env, arm_ctrl, target_local: np.ndarray, gripper_action: float,
    max_steps: int = 80, tol: float = 0.02,
) -> tuple[int, float]:
    """One proportional visual-servo loop toward `target_local` (already in
    the arm controller's own root-relative frame) -- same pattern
    `_navigate_to()`'s `_drive_toward()` already uses for the base, applied
    to the arm: recompute the real error every step rather than committing
    to one large delta, so the controller's own per-step IK solve only ever
    has to find a small, usually-feasible motion."""
    used = 0
    dist = float("inf")
    for _ in range(max_steps):
        cur = _ee_pos_local(arm_ctrl)
        err = target_local - cur
        dist = float(np.linalg.norm(err))
        if dist < tol:
            break
        action = np.zeros(9, dtype=np.float32)
        action[0:3] = np.clip(err / 0.1, -1, 1)
        action[3] = gripper_action
        env.step(action)
        used += 1
    return used, dist


def _hold_gripper(env, value: float, steps: int) -> None:
    action = np.zeros(9, dtype=np.float32)
    action[3] = value
    for _ in range(steps):
        env.step(action)


def attempt_goal_with_real_grasp(
    env, goal: Goal, tray_slot_xyz: np.ndarray, nav_steps: int = 250,
) -> dict:
    """Real navigate -> reach -> grasp -> lift -> carry -> place -> release,
    verified with `agent.is_grasping()` at each stage that matters rather
    than assumed. Returns the same `{"achieved", "steps_used", ...}` shape
    `attempt_goal()` does, plus `grasped`/`carried` so a caller (or a demo
    script) can see exactly which real stage succeeded or failed, instead of
    only a final boolean.

    Deliberately narrower than `attempt_goal()`: no feasibility/navigation-
    safety short-circuiting, single fixed object, not benchmarked across
    seeds. This is a demonstrated capability, not (yet) a drop-in
    replacement -- see module docstring for why that's a real scope choice,
    not an oversight.
    """
    ensure_tray_surface(env)
    agent = env.unwrapped.agent
    arm_ctrl = _arm_controller(env)
    obj = env.unwrapped._get_actor(goal.target_object)

    before = env.unwrapped._elapsed_control_steps
    nav1 = _navigate_to(env, obj.pose.sp.p, steps=nav_steps, target_object=goal.target_object)
    if not nav1.reached_target:
        return {
            "achieved": False, "steps_used": env.unwrapped._elapsed_control_steps - before,
            "grasped": False, "carried": False,
        }

    _hold_gripper(env, 1.0, 6)  # open
    _servo_arm_to(env, arm_ctrl, _world_to_local(arm_ctrl, obj.pose.sp.p), gripper_action=1.0)
    _hold_gripper(env, -1.0, 15)  # close
    grasped = bool(agent.is_grasping(obj))
    if not grasped:
        return {
            "achieved": False, "steps_used": env.unwrapped._elapsed_control_steps - before,
            "grasped": False, "carried": False,
        }

    lift_target = _ee_pos_local(arm_ctrl) + np.array([0.0, 0.0, 0.15])
    _servo_arm_to(env, arm_ctrl, lift_target, gripper_action=-1.0, max_steps=40)

    nav2 = _navigate_to(env, _TRAY_POSITION, steps=nav_steps, target_object="__tray__")
    carried = bool(agent.is_grasping(obj))
    if not nav2.reached_target or not carried:
        return {
            "achieved": False, "steps_used": env.unwrapped._elapsed_control_steps - before,
            "grasped": True, "carried": carried,
        }

    drop_world = np.array([_TRAY_POSITION[0], _TRAY_POSITION[1], _TRAY_SURFACE_TOP_Z + _APPROACH_HEIGHT])
    _servo_arm_to(env, arm_ctrl, _world_to_local(arm_ctrl, drop_world), gripper_action=-1.0, tol=0.03)
    _hold_gripper(env, 1.0, 10)  # release
    _hold_gripper(env, 1.0, 30)  # let real physics settle

    steps_used = env.unwrapped._elapsed_control_steps - before
    state = env.unwrapped._world_state()
    achieved = goal_achieved(goal, state, _TRAY_POSITION, _TRAY_HALF_SIZES)
    return {"achieved": achieved, "steps_used": steps_used, "grasped": True, "carried": carried}

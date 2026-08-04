"""D-028 (ai-notes/decisions.md): a proper analytic-Jacobian IK solver for
G1's right arm, built to retry D-024's real contact/tactile grasp
confirmation with a principled tool instead of the finite-difference
approximation that proved unreliable there (same starting state converged
to wildly different distances run to run).

Promoted to src/atr/ 2026-08-04 (D-051) -- see ai-notes/decisions.md.
Was already zero-dependency on any other spike file (only `numpy`,
`pinocchio`, `mani_skill.PACKAGE_ASSET_DIR`), so promotion was a plain
`git mv`, nothing to fix.

Uses `pinocchio` (already a dependency, proven for Panda's Cartesian
controller in the original spike) against G1's actual URDF kinematic chain
-- a real analytic Jacobian, not a numerical approximation. Damped
least-squares (Levenberg-Marquardt-style), which is standard for this and
numerically stable near singularities, unlike a plain pseudo-inverse.

Verified before trusting it for anything: pinocchio's forward kinematics
for `right_tcp_link`, evaluated in the URDF's own local frame, matches
ManiSkill's `agent.right_tcp.pose.sp.p - agent.robot.pose.sp.p` (world tcp
minus world base) to 5 decimal places -- G1's base has zero rotation when
placed via `sapien.Pose(p=...)`, so world-to-local is a plain translation,
confirmed rather than assumed.

Result, not just tooling: this solver is deterministic (identical distance
across repeated runs, unlike the finite-difference version) and, searched
across a wide floor-clearance-checked set of candidate base positions with
random-restart initialization, cannot bring G1's right tcp within ~13cm of
either potted_meat_can or master_chef_can in the "kitchen_cabinet" scene.
Not joint-limit-bound (checked directly). This is a real kinematic
reachability limit of this arm, at this scene, for these objects -- not a
solver artifact, and not fixable by better solver tuning. See
`test_ik_solver.py`'s `TestConfirmedUnreachable` for the regression test
that locks this finding in, so it isn't silently re-litigated later.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import pinocchio as pin
from mani_skill import PACKAGE_ASSET_DIR

_URDF_PATH = PACKAGE_ASSET_DIR / "robots/g1_humanoid/g1_simplified_upper_body.urdf"

# Matches agent.body_joints order exactly (both come from the same URDF) --
# verified directly, not assumed: pinocchio's joint order after
# buildModelFromUrdf matched ManiSkill's body_joints list index-for-index.
_ARM_JOINT_NAMES = (
    "torso_joint", "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint", "right_elbow_pitch_joint", "right_elbow_roll_joint",
)


@lru_cache(maxsize=1)
def _model_and_data():
    model = pin.buildModelFromUrdf(str(_URDF_PATH))
    return model, model.createData()


def solve_right_arm_ik(
    target_local: np.ndarray,
    body_joint_names: list[str],
    qpos_init: np.ndarray | None = None,
    damping: float = 1e-2,
    max_iters: int = 150,
    tol: float = 0.01,
) -> tuple[np.ndarray, float]:
    """Damped-least-squares IK for G1's right_tcp_link, in the robot's own
    base-local frame (see module docstring for the local-frame convention).

    `target_local` = target_world_position - robot_base_world_position
    (zero rotation assumed for the base, verified in the module docstring).
    `body_joint_names` must be agent.body_joints from the live ManiSkill
    agent, so the returned qpos indexes correctly into it.

    Returns (qpos, final_distance_to_target). Only moves the 6 right-arm
    joints (_ARM_JOINT_NAMES); everything else stays at qpos_init.
    """
    model, data = _model_and_data()
    frame_id = model.getFrameId("right_tcp_link")
    idx = {name: i for i, name in enumerate(body_joint_names)}
    arm_idx = [idx[name] for name in _ARM_JOINT_NAMES]
    lower, upper = model.lowerPositionLimit, model.upperPositionLimit

    qpos = np.zeros(model.nq) if qpos_init is None else qpos_init.copy()
    dist = float("inf")
    for _ in range(max_iters):
        pin.forwardKinematics(model, data, qpos)
        pin.updateFramePlacements(model, data)
        current = data.oMf[frame_id].translation
        error = target_local - current
        dist = float(np.linalg.norm(error))
        if dist < tol:
            break
        jacobian = pin.computeFrameJacobian(model, data, qpos, frame_id, pin.LOCAL_WORLD_ALIGNED)[:3, arm_idx]
        delta = jacobian.T @ np.linalg.solve(jacobian @ jacobian.T + damping * np.eye(3), error)
        qpos[arm_idx] = np.clip(qpos[arm_idx] + np.clip(delta, -0.2, 0.2), lower[arm_idx], upper[arm_idx])
    return qpos, dist


def best_reachable_distance(
    target_local: np.ndarray, body_joint_names: list[str], n_restarts: int = 10, seed: int = 0,
) -> float:
    """Random-restart wrapper: returns the closest distance found across
    `n_restarts` different starting configurations (all-zero plus random
    joint angles), since a single damped-least-squares run can settle into
    a local minimum that isn't the arm's true closest approach."""
    model, _ = _model_and_data()
    idx = {name: i for i, name in enumerate(body_joint_names)}
    arm_idx = [idx[name] for name in _ARM_JOINT_NAMES]
    lower, upper = model.lowerPositionLimit, model.upperPositionLimit
    rng = np.random.default_rng(seed)

    best = float("inf")
    for trial in range(n_restarts):
        qpos_init = np.zeros(model.nq)
        if trial > 0:
            for ai in arm_idx:
                qpos_init[ai] = rng.uniform(lower[ai], upper[ai])
        _, dist = solve_right_arm_ik(target_local, body_joint_names, qpos_init=qpos_init)
        best = min(best, dist)
    return best

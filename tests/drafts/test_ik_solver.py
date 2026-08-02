"""Tests for ik_solver.py -- D-028's retry of D-024's real contact/tactile
grasp confirmation, using a proper analytic-Jacobian IK solver instead of
the finite-difference approximation that proved unreliable there.
"""

import gymnasium as gym
import numpy as np
import pytest

pytest.importorskip("mani_skill")
pytest.importorskip("pinocchio")

import task_schema_draft  # noqa: E402, F401
from task_schema_draft.ik_solver import best_reachable_distance, solve_right_arm_ik  # noqa: E402


def _make_env():
    return gym.make(
        "TidyUpTaskSchemaDraft-ReplicaCAD-Humanoid-v1", num_envs=1, obs_mode="state",
        render_mode=None, sim_backend="physx_cpu", control_mode="pd_joint_pos",
        intervention_kind="none",
    )


class TestMatchesManiSkillKinematics:
    """Real verification, not an assumption: pinocchio's local-frame FK for
    right_tcp_link must agree with ManiSkill's own world-frame tcp position
    (minus the robot's world base position) before trusting this solver for
    anything -- see module docstring."""

    def test_local_frame_fk_matches_maniskill_world_tcp(self):
        env = _make_env()
        try:
            env.reset(seed=0)
            agent = env.unwrapped.agent
            base_pos = agent.robot.pose.sp.p.copy()
            qpos = agent.robot.qpos[0].cpu().numpy().astype(np.float64)

            from task_schema_draft.ik_solver import _model_and_data

            model, data = _model_and_data()
            import pinocchio as pin

            pin.forwardKinematics(model, data, qpos)
            pin.updateFramePlacements(model, data)
            pinocchio_local = data.oMf[model.getFrameId("right_tcp_link")].translation

            maniskill_local = agent.right_tcp.pose.sp.p - base_pos
            assert np.allclose(pinocchio_local, maniskill_local, atol=1e-4)
        finally:
            env.close()


class TestDeterministic:
    """The finite-difference solver D-024 originally tried gave 11cm on one
    run and 57cm on another for identical inputs -- this is the property
    that made it untrustworthy. This solver must not do that."""

    def test_same_input_gives_same_output_every_time(self):
        target = np.array([-0.258, -0.141, -0.076])
        body_joints = [
            "torso_joint", "left_shoulder_pitch_joint", "right_shoulder_pitch_joint",
            "left_shoulder_roll_joint", "right_shoulder_roll_joint", "left_shoulder_yaw_joint",
            "right_shoulder_yaw_joint", "left_elbow_pitch_joint", "right_elbow_pitch_joint",
            "left_elbow_roll_joint", "right_elbow_roll_joint", "left_zero_joint", "left_three_joint",
            "left_five_joint", "right_zero_joint", "right_three_joint", "right_five_joint",
            "left_one_joint", "left_four_joint", "left_six_joint", "right_one_joint",
            "right_four_joint", "right_six_joint", "left_two_joint", "right_two_joint",
        ]
        distances = [solve_right_arm_ik(target, body_joints)[1] for _ in range(5)]
        assert len(set(round(d, 8) for d in distances)) == 1, f"non-deterministic: {distances}"


class TestConfirmedUnreachable:
    """Locks in D-028's actual finding so it isn't silently re-litigated:
    neither target object can be brought within real contact range of
    G1's right_tcp_link from the "kitchen_cabinet" scene's base position,
    confirmed via random-restart search across this solver (deterministic,
    matches real kinematics) -- not a solver artifact. A physical grasp
    (finger closure generating real contact force) needs roughly <5cm;
    both objects plateau at >10cm regardless of restart."""

    def test_potted_meat_can_not_within_contact_range(self):
        env = _make_env()
        try:
            env.reset(seed=0)
            agent = env.unwrapped.agent
            base_pos = agent.robot.pose.sp.p.copy()
            target_local = env.unwrapped._get_actor("potted_meat_can").pose.sp.p.copy() - base_pos
            best = best_reachable_distance(target_local, agent.body_joints, n_restarts=6)
        finally:
            env.close()
        assert best > 0.10, f"expected still-unreachable (>10cm), got {best:.4f}m -- re-check D-028's finding"

    def test_master_chef_can_not_within_contact_range(self):
        env = _make_env()
        try:
            env.reset(seed=0)
            agent = env.unwrapped.agent
            base_pos = agent.robot.pose.sp.p.copy()
            target_local = env.unwrapped._get_actor("master_chef_can").pose.sp.p.copy() - base_pos
            best = best_reachable_distance(target_local, agent.body_joints, n_restarts=6)
        finally:
            env.close()
        assert best > 0.10, f"expected still-unreachable (>10cm), got {best:.4f}m -- re-check D-028's finding"

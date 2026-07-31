"""Tests for the G1-placed-in-the-real-apartment version of TidyUp — the
direct answer to "but this is not a humanoid robot" for the ReplicaCAD
variant. Same goal_graph/oracle_feasibility/intent_guard, no navigation
(G1 is fixed-base). See tidy_up_env_replicacad_humanoid.py's module
docstring and ../README.md "G1 in the real apartment" for what had to be
fixed and why.
"""

import gymnasium as gym
import pytest

pytest.importorskip("mani_skill")

import task_schema_draft  # noqa: E402, F401  (registers TidyUpTaskSchemaDraft-ReplicaCAD-Humanoid-v1)
from task_schema_draft.policy_baselines_replicacad_humanoid import (  # noqa: E402
    feasibility_aware_policy,
    naive_substitution_policy,
    static_policy,
)


def _make_env(**kwargs):
    return gym.make(
        "TidyUpTaskSchemaDraft-ReplicaCAD-Humanoid-v1", num_envs=1, obs_mode="state",
        render_mode=None, sim_backend="physx_cpu", control_mode="pd_joint_pos", **kwargs,
    )


class TestTidyUpReplicaCADHumanoidEnv:
    def test_registered(self):
        assert "TidyUpTaskSchemaDraft-ReplicaCAD-Humanoid-v1" in gym.envs.registry

    def test_reset_and_step(self):
        env = _make_env(intervention_kind="none")
        try:
            obs, info = env.reset(seed=0)
            assert "goal_feasibility" in info
            env.step(env.action_space.sample() * 0)
        finally:
            env.close()

    def test_objects_are_placed_at_real_positions_not_floating(self):
        """Regression test for the actual bug found: catching the scene
        builder's fetch-only NotImplementedError skipped its second
        object-placement pass, leaving every object ~1000m in the air.
        The fix (temporarily presenting as "fetch" during initialization)
        must result in objects at their real height."""
        env = _make_env(intervention_kind="none")
        try:
            env.reset(seed=0)
            for alias in ("potted_meat_can", "master_chef_can", "bowl", "cracker_box"):
                z = env.unwrapped._get_actor(alias).pose.sp.p[2]
                assert abs(z) < 5, f"{alias} is at z={z}, looks like it's still floating"
        finally:
            env.close()

    def test_scene_layout_reproducible_across_seeds(self):
        """Regression test for the bug D-020's vision work found: object
        positions used to depend on the `seed` passed to reset() (via
        torch's global RNG, not this env's own _episode_rng) -- seed=2
        loaded a different apartment entirely, and some seeds hid target
        objects at z=-10000 outright, even with build_config_idxs pinned
        alone. G1's base pose, camera, and vision.py's crops are only valid
        for one specific layout, so this must hold regardless of seed."""
        positions_by_seed = {}
        for seed in (0, 2, 15, 42):
            env = _make_env(intervention_kind="none")
            try:
                env.reset(seed=seed)
                positions_by_seed[seed] = {
                    alias: tuple(env.unwrapped._get_actor(alias).pose.sp.p.tolist())
                    for alias in ("potted_meat_can", "master_chef_can", "bowl", "cracker_box")
                }
            finally:
                env.close()
        reference = positions_by_seed[0]
        for seed, positions in positions_by_seed.items():
            assert positions == reference, f"seed={seed} layout differs from seed=0: {positions}"

    def test_no_false_violations_from_settling(self):
        env = _make_env(intervention_kind="none")
        try:
            env.reset(seed=0)
            for _ in range(15):
                _, _, _, _, info = env.step(env.action_space.sample() * 0)
            assert not any(info["constraint_violations"].values())
        finally:
            env.close()


class TestReplicaCADHumanoidPolicyComparison:
    def test_static_vs_feasibility_aware_same_recall_less_waste(self):
        results = {}
        for name, policy in [("static", static_policy), ("feasibility_aware", feasibility_aware_policy)]:
            env = _make_env(intervention_kind="chef_can_destroyed", onset_step_range=(2, 3))
            try:
                env.reset(seed=0)
                results[name] = policy(env)
            finally:
                env.close()
        assert results["static"]["goals_achieved"] == results["feasibility_aware"]["goals_achieved"]
        assert results["feasibility_aware"]["wasted_steps"] == 0
        assert results["static"]["wasted_steps"] > 0

    def test_intent_guard_blocks_substitution_without_recall_cost(self):
        results = {}
        for guarded in (False, True):
            env = _make_env(intervention_kind="chef_can_destroyed", onset_step_range=(2, 3))
            try:
                env.reset(seed=0)
                results[guarded] = naive_substitution_policy(env, use_intent_guard=guarded)
            finally:
                env.close()
        assert results[False]["dont_move_bowl_violated"] is True
        assert results[True]["dont_move_bowl_violated"] is False
        assert results[False]["goals_achieved"] == results[True]["goals_achieved"]

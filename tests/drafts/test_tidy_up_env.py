"""End-to-end tests for TidyUpEnv wiring the goal graph + oracle feasibility
to a real ManiSkill3 scene. See spikes/task_schema_draft/README.md.
"""

import pytest

pytest.importorskip("mani_skill")

import gymnasium as gym  # noqa: E402

import task_schema_draft  # noqa: E402, F401  (registers TidyUp-v1)


def _make_env(**kwargs):
    # Always CPU — object add/remove is unsupported under GPU-batched sim
    # (see tidy_up_env.py's module docstring).
    return gym.make(
        "TidyUp-v1",
        num_envs=1,
        obs_mode="state",
        render_mode=None,
        sim_backend="physx_cpu",
        **kwargs,
    )


class TestBowlDestroyedIntervention:
    def test_bowl_goal_infeasible_after_destruction(self):
        env = _make_env(intervention_kind="bowl_destroyed", onset_step_range=(3, 4))
        try:
            env.reset(seed=0)
            for _ in range(3):
                _, _, _, _, info = env.step(env.action_space.sample() * 0)
            assert info["goal_feasibility"]["place_blue_bowl"] is True  # not yet triggered
            _, _, _, _, info = env.step(env.action_space.sample() * 0)  # step 3: triggers
            assert info["goal_feasibility"]["place_blue_bowl"] is False
            assert info["goal_feasibility"]["place_red_mug"] is True
            assert not any(info["constraint_violations"].values())
        finally:
            env.close()

    def test_reproducible_given_seed(self):
        onset_steps_seen = []
        for _ in range(2):
            env = _make_env(intervention_kind="bowl_destroyed")
            try:
                env.reset(seed=7)
                for step in range(30):
                    _, _, _, _, info = env.step(env.action_space.sample() * 0)
                    if info["goal_feasibility"]["place_blue_bowl"] is False:
                        onset_steps_seen.append(step)
                        break
            finally:
                env.close()
        assert len(onset_steps_seen) == 2
        assert onset_steps_seen[0] == onset_steps_seen[1]


class TestTemporaryObstacleIntervention:
    def test_world_change_never_flips_feasibility(self):
        """The matched reversible/temporary control from docs/04: a
        detectable world change (obstacle appears, then disappears) that
        never makes any goal infeasible or violates a constraint."""
        env = _make_env(
            intervention_kind="temporary_obstacle", onset_step_range=(3, 4),
            obstacle_duration_steps=4,
        )
        try:
            env.reset(seed=0)
            saw_obstacle = False
            for _ in range(12):
                _, _, _, _, info = env.step(env.action_space.sample() * 0)
                saw_obstacle = saw_obstacle or bool(info["obstacle_present"])
                assert all(info["goal_feasibility"].values())
                assert not any(info["constraint_violations"].values())
            assert saw_obstacle, "obstacle should have appeared at some point"
            assert not info["obstacle_present"], "obstacle should be gone again by the end"
        finally:
            env.close()


class TestNoIntervention:
    def test_baseline_stays_fully_feasible(self):
        env = _make_env(intervention_kind="none")
        try:
            env.reset(seed=0)
            for _ in range(10):
                _, _, _, _, info = env.step(env.action_space.sample() * 0)
            assert all(info["goal_feasibility"].values())
            assert not any(info["constraint_violations"].values())
        finally:
            env.close()

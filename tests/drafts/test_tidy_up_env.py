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


class TestResourceContentionIntervention:
    """D-059: a third, mechanistically different intervention (not a blind
    timer like bowl_destroyed) -- blue_bowl is only taken if the agent
    hasn't already secured it (placed it on the tray) by the onset step.
    Exists to unlock a real held-out-intervention split (only 2 kinds
    existed before this), matched with a reversible counterpart per
    docs/04, same as bowl_destroyed/temporary_obstacle already are."""

    def test_bowl_not_yet_secured_becomes_infeasible_at_onset(self):
        env = _make_env(intervention_kind="resource_contention", onset_step_range=(3, 4))
        try:
            env.reset(seed=0)
            for _ in range(3):
                _, _, _, _, info = env.step(env.action_space.sample() * 0)
            assert info["goal_feasibility"]["place_blue_bowl"] is True  # not yet triggered
            _, _, _, _, info = env.step(env.action_space.sample() * 0)  # step 3: triggers
            assert info["goal_feasibility"]["place_blue_bowl"] is False
            assert info["goal_feasibility"]["place_red_mug"] is True
        finally:
            env.close()

    def test_bowl_already_secured_before_onset_is_never_taken(self):
        """The real behavioral difference from bowl_destroyed: this
        intervention checks episode progress, not just elapsed time -- a
        policy that secures the contested resource in time never sees it
        fire at all, even past the onset step."""
        import sapien

        env = _make_env(intervention_kind="resource_contention", onset_step_range=(3, 4))
        try:
            env.reset(seed=0)
            env.unwrapped._objects["blue_bowl"].set_pose(sapien.Pose(p=[0.4, 0.0, 0.005]))
            for _ in range(8):
                _, _, _, _, info = env.step(env.action_space.sample() * 0)
            assert info["goal_feasibility"]["place_blue_bowl"] is True
            assert env.unwrapped._exists["blue_bowl"] is True
        finally:
            env.close()

    def test_temporary_variant_recovers_feasibility_after_the_resource_returns(self):
        env = _make_env(
            intervention_kind="resource_contention_temporary", onset_step_range=(3, 4),
            obstacle_duration_steps=3,
        )
        try:
            env.reset(seed=0)
            feasibility_over_time = []
            for _ in range(10):
                _, _, _, _, info = env.step(env.action_space.sample() * 0)
                feasibility_over_time.append(info["goal_feasibility"]["place_blue_bowl"])
            assert False in feasibility_over_time, "should have gone infeasible at some point"
            assert feasibility_over_time[-1] is True, "should have come back by the end"
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

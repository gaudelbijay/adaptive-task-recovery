"""Tests for the ReplicaCAD + Fetch version of TidyUp — a real furnished
apartment scene (Habitat/ManiSkill3's own ReplicaCADSetTableTrain builder)
with real YCB objects and a mobile robot, instead of a hand-built scene.
Same goal_graph/oracle_feasibility/intent_guard as the panda and humanoid
variants. See tidy_up_env_replicacad.py and navigation.py module docstrings,
and ../README.md "ReplicaCAD embodiment" for what had to change and why.

Requires the ReplicaCAD + ReplicaCADRearrange + ycb asset downloads (see
README "How to run it").
"""

import numpy as np
import pytest

pytest.importorskip("mani_skill")

import gymnasium as gym  # noqa: E402

import task_schema_draft  # noqa: E402, F401  (registers TidyUp-ReplicaCAD-v1)
from atr.envs.navigation import build_occupancy_grid, plan_path  # noqa: E402
from atr.envs.tidy_up_replicacad_policies import (  # noqa: E402
    feasibility_aware_policy,
    naive_substitution_policy,
    static_policy,
)


def _make_env(**kwargs):
    return gym.make(
        "TidyUp-ReplicaCAD-v1", num_envs=1, obs_mode="state", render_mode=None,
        sim_backend="physx_cpu", control_mode="pd_ee_delta_pos", **kwargs,
    )


class TestTidyUpReplicaCADEnv:
    def test_registered(self):
        assert "TidyUp-ReplicaCAD-v1" in gym.envs.registry

    def test_reset_and_step(self):
        env = _make_env(intervention_kind="none")
        try:
            obs, info = env.reset(seed=0)
            assert "goal_feasibility" in info
            env.step(env.action_space.sample() * 0)
        finally:
            env.close()

    def test_scene_layout_reproducible_across_seeds(self):
        """Same bug/fix as tidy_up_env_replicacad_humanoid.py's test of the
        same name -- this env shares the same scene_builder_cls, and was
        confirmed to have the same bug: seed=2 hid both potted_meat_can and
        bowl at z=-10000, this env's own two goal objects."""
        positions_by_seed = {}
        for seed in (0, 2, 7, 42):
            env = _make_env(intervention_kind="none")
            try:
                env.reset(seed=seed)
                positions_by_seed[seed] = {
                    alias: tuple(env.unwrapped._get_actor(alias).pose.sp.p.tolist())
                    for alias in ("potted_meat_can", "bowl", "master_chef_can", "cracker_box")
                }
            finally:
                env.close()
        reference = positions_by_seed[0]
        for seed, positions in positions_by_seed.items():
            assert positions == reference, f"seed={seed} layout differs from seed=0: {positions}"


class TestNavigation:
    def test_path_planner_routes_around_the_real_wall_that_blocked_naive_control(self):
        """Regression test for the actual bug found: a naive point-and-drive
        controller got physically stuck against a real wall/doorway in this
        scene (confirmed via raycast at the time). The planner must find a
        path, and that path must not be a straight line through the
        obstacle."""
        env = _make_env(intervention_kind="none")
        try:
            env.reset(seed=0)
            px = env.unwrapped.scene.px
            xs, ys, occupied = build_occupancy_grid(
                px, (-2.5, 1.0), (-1.5, 1.5), robot_radius=0.2
            )
            start = env.unwrapped.agent.base_link.pose.sp.p[:2]
            path = plan_path(xs, ys, occupied, start, np.array([0.29, 0.09]))
            assert path is not None
            assert len(path) > 2  # more than one straight hop -- a real detour
        finally:
            env.close()


class TestReplicaCADPolicyComparison:
    def test_static_vs_feasibility_aware_same_recall_less_waste(self):
        results = {}
        for name, policy in [("static", static_policy), ("feasibility_aware", feasibility_aware_policy)]:
            env = _make_env(intervention_kind="bowl_destroyed", onset_step_range=(2, 3))
            try:
                env.reset(seed=0)
                results[name] = policy(env)
            finally:
                env.close()
        assert results["static"]["goals_achieved"] == results["feasibility_aware"]["goals_achieved"]
        assert results["feasibility_aware"]["wasted_steps"] == 0
        assert results["static"]["wasted_steps"] > 0

    def test_intent_guard_blocks_substitution_without_recall_cost(self):
        """D-105: this must protect `cracker_box`, not the default graph's
        `master_chef_can`. Confirmed directly (not assumed): `plan_path()`
        cannot route to `master_chef_can`'s real resting position from Fetch's
        spawn in this scene at all -- the grid genuinely disconnects that
        region from the rest of the apartment (166 connected components;
        start and goal land in different ones; confirmed not a discretization
        artifact by sweeping resolution from 0.15 down to 0.05 with no
        change). That's invisible everywhere else in this project because
        `master_chef_can` is only ever a *protected* object other tests
        navigate around (D-096--D-104), never a real navigation target --
        this ablation is the one place that asks Fetch to travel all the way
        to it. `cracker_box` is confirmed reachable (`plan_path()` finds a
        route), so swapping the protected object via a custom `GoalGraph`
        (D-100's own pattern) restores the test's original, full-strength
        claim -- a real physical violation without the guard -- instead of
        weakening it to something the physical scene can't actually
        demonstrate for this specific object."""
        from atr.language.goal_graph import Constraint, GoalGraph
        from atr.envs.tidy_up_env_replicacad import replicacad_example

        base_graph = replicacad_example()
        graph = GoalGraph(
            instruction_text=(
                "Put the potted meat can and bowl on the table, and do not "
                "move the cracker box."
            ),
            goals=base_graph.goals,
            constraints=(
                Constraint(
                    id="dont_move_cracker_box", kind="never_move",
                    target_object="cracker_box", tolerance=0.05,
                ),
            ),
        )
        results = {}
        for guarded in (False, True):
            env = _make_env(intervention_kind="bowl_destroyed", onset_step_range=(2, 3))
            try:
                env.reset(seed=0)
                results[guarded] = naive_substitution_policy(env, graph=graph, use_intent_guard=guarded)
            finally:
                env.close()
        assert results[False]["dont_move_cracker_box_violated"] is True
        assert results[True]["dont_move_cracker_box_violated"] is False
        assert results[False]["goals_achieved"] == results[True]["goals_achieved"]


class TestReplicaCADMultiSeedBenchmark:
    """D-108: `test_static_vs_feasibility_aware_same_recall_less_waste` above
    is a single seed with a narrow onset window (2-3) -- it shows the
    direction of the effect but not whether it holds under real seed
    variance. A variance sweep (`onset_step_range=(20, 500)`) found the
    narrow window degenerate (zero wasted steps, identical outcome, every
    seed) while the wider one produces a real mix of outcomes across seeds.
    This exercises D-091-107's navigation-safety machinery -- previously
    validated only on hand-placed single scenarios -- under genuine seed
    variance for the first time, using the project's own paired bootstrap
    protocol (docs/10)."""

    def test_oracle_feasibility_matches_static_recall_and_wastes_fewer_steps(self):
        from atr.evaluation.harness import bootstrap_ci

        seeds = list(range(30))
        per_seed = {"static": [], "oracle_feasibility": []}
        for seed in seeds:
            for name, policy in [
                ("static", static_policy), ("oracle_feasibility", feasibility_aware_policy),
            ]:
                env = _make_env(intervention_kind="bowl_destroyed", onset_step_range=(20, 500))
                try:
                    env.reset(seed=seed)
                    per_seed[name].append(policy(env))
                finally:
                    env.close()

        goals_static = [r["goals_achieved"] for r in per_seed["static"]]
        goals_oracle = [r["goals_achieved"] for r in per_seed["oracle_feasibility"]]
        # Paired per seed: feasibility awareness changes *how* goals are
        # pursued, not *which* ones are achievable, so recall must match
        # exactly seed-for-seed, not just in aggregate.
        assert goals_static == goals_oracle

        wasted_static = [r["wasted_steps"] for r in per_seed["static"]]
        wasted_oracle = [r["wasted_steps"] for r in per_seed["oracle_feasibility"]]
        diff = [s - o for s, o in zip(wasted_static, wasted_oracle)]
        _, lo, _ = bootstrap_ci(diff)
        # Paired bootstrap on the per-seed difference, not a naive overlap
        # check on the two policies' independent CIs -- those do overlap
        # (measured: static 161.7 [123.2, 200.2] vs oracle 115.5 [77.0,
        # 154.0]) even though the paired per-seed effect is real.
        assert lo > 0

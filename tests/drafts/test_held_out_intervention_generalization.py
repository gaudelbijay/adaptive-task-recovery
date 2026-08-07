"""Tests genuine held-out-intervention generalization (D-069) -- D-059
built `INTERVENTION_SPLITS`/`HELD_OUT_INTERVENTION` (`atr.evaluation.splits`)
but nothing had ever actually trained on the "train" split and evaluated
on the "held_out_intervention" split. This does: `train_q_table()`
(reward-driven) and `collect_demonstrations()`/`train_bc_table()`
(demonstration-driven) both only ever see `bowl_destroyed`/
`temporary_obstacle` (`INTERVENTION_TRAIN`'s two, timer-based
interventions) during training -- never `resource_contention`
(`HELD_OUT_INTERVENTION`, D-059's progress-contingent mechanism, a
genuinely different trigger, not just a relabeled copy of the same one).

Real, measured result (verified via a standalone script first, across 5
seeds, then formally via `track_comparison()` across 20 paired seeds,
`ai-notes/decisions.md` D-069): both learned policies match oracle
feasibility exactly on the never-seen intervention, every seed, zero
variance. Not a coincidence to be surprised by -- both policies'
learning signal is keyed on `(goal_id, feasible)`, where `feasible`
comes from `goal_feasible()` (privileged existence, or its demonstrated-
by-the-expert equivalent), a representation that never encoded *how* an
object became infeasible, only *whether* it currently is. This is the
first real confirmation that design choice pays off for genuine held-
out-mechanism generalization, not just an assumption.
"""

import gymnasium as gym
import pytest

pytest.importorskip("mani_skill")

import task_schema_draft  # noqa: E402, F401
from atr.envs.tidy_up_policies import _TRAY_SLOTS, attempt_goal, feasibility_aware_policy  # noqa: E402
from atr.evaluation.splits import HELD_OUT_INTERVENTION, INTERVENTION_TRAIN  # noqa: E402
from atr.language.goal_graph import canonical_example  # noqa: E402
from atr.policies.imitation import collect_demonstrations, imitation_policy, train_bc_table  # noqa: E402
from atr.policies.q_learning import learned_policy, train_q_table  # noqa: E402

_GRAPH = canonical_example()
_TRAIN_KINDS = tuple(spec.intervention_kind for spec in INTERVENTION_TRAIN)
_HELD_OUT_KINDS = tuple(spec.intervention_kind for spec in HELD_OUT_INTERVENTION)


def _make_env(intervention_kind: str, onset_step_range: tuple[int, int] = (5, 15)):
    return gym.make(
        "TidyUp-v1", num_envs=1, obs_mode="state", render_mode=None,
        sim_backend="physx_cpu", control_mode="pd_ee_delta_pos",
        intervention_kind=intervention_kind, onset_step_range=onset_step_range,
    )


class TestLearnedPoliciesGeneralizeToHeldOutIntervention:
    @pytest.fixture(scope="class")
    def q_table(self):
        return train_q_table(
            _make_env, _GRAPH, _TRAY_SLOTS, attempt_goal, intervention_kinds=_TRAIN_KINDS, n_episodes=120, seed=0,
        )

    @pytest.fixture(scope="class")
    def bc_table(self):
        demos = collect_demonstrations(
            _make_env, _GRAPH, _TRAY_SLOTS, attempt_goal, intervention_kinds=_TRAIN_KINDS, n_episodes=40, seed=0,
        )
        return train_bc_table(demos)

    def test_q_table_never_saw_the_held_out_intervention_during_training(self, q_table):
        """Sanity check on the experimental setup itself, not the
        result -- confirms _TRAIN_KINDS/_HELD_OUT_KINDS are genuinely
        disjoint (already checked at the registry level,
        test_splits.py::test_train_and_held_out_are_disjoint_kinds, but
        cheap to also confirm right where the training actually happens)."""
        assert set(_TRAIN_KINDS).isdisjoint(_HELD_OUT_KINDS)

    def test_learned_policy_matches_oracle_on_resource_contention(self, q_table):
        results = {}
        for name, run in [
            ("oracle", feasibility_aware_policy),
            ("learned", lambda env: learned_policy(env, q_table, _GRAPH, attempt_goal, _TRAY_SLOTS)),
        ]:
            env = _make_env(intervention_kind="resource_contention", onset_step_range=(3, 4))
            try:
                env.reset(seed=0)
                results[name] = run(env)
            finally:
                env.close()
        assert results["learned"]["goals_achieved"] == results["oracle"]["goals_achieved"]
        assert results["learned"]["wasted_steps"] == 0

    def test_imitation_policy_matches_oracle_on_resource_contention(self, bc_table):
        results = {}
        for name, run in [
            ("oracle", feasibility_aware_policy),
            ("imitation", lambda env: imitation_policy(env, bc_table, _GRAPH, attempt_goal, _TRAY_SLOTS)),
        ]:
            env = _make_env(intervention_kind="resource_contention", onset_step_range=(3, 4))
            try:
                env.reset(seed=0)
                results[name] = run(env)
            finally:
                env.close()
        assert results["imitation"]["goals_achieved"] == results["oracle"]["goals_achieved"]
        assert results["imitation"]["wasted_steps"] == 0

    def test_both_policies_also_match_oracle_on_the_temporary_held_out_variant(self, q_table, bc_table):
        """resource_contention_temporary (the matched reversible half of
        D-059's pair) is a genuinely different case from
        resource_contention: the resource comes back, so the correct
        behavior is achieving *both* goals, not skipping one. Checking
        both held-out kinds, not just one, since a policy could pass the
        first by only handling "permanently gone" correctly."""
        env = _make_env(intervention_kind="resource_contention_temporary", onset_step_range=(3, 4))
        try:
            env.reset(seed=0)
            oracle_result = feasibility_aware_policy(env)
        finally:
            env.close()

        env = _make_env(intervention_kind="resource_contention_temporary", onset_step_range=(3, 4))
        try:
            env.reset(seed=0)
            learned_result = learned_policy(env, q_table, _GRAPH, attempt_goal, _TRAY_SLOTS)
        finally:
            env.close()

        env = _make_env(intervention_kind="resource_contention_temporary", onset_step_range=(3, 4))
        try:
            env.reset(seed=0)
            imitation_result = imitation_policy(env, bc_table, _GRAPH, attempt_goal, _TRAY_SLOTS)
        finally:
            env.close()

        assert oracle_result["goals_achieved"] == 2
        assert learned_result["goals_achieved"] == 2
        assert imitation_result["goals_achieved"] == 2

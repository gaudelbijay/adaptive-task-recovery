"""Tests for atr.policies.imitation (D-060) -- imitation learning
(behavioral cloning) over the same goal-attempt decision
atr.policies.q_learning learns via reward, so the two can be compared
under genuinely matched conditions. See imitation.py's module docstring
for what this is meant to demonstrate: given demonstration coverage
comparable to what Q-learning explores, imitation matches it; given
narrower demonstration coverage, imitation inherits that narrowness in a
way Q-learning's own exploration doesn't.

No rendering anywhere here (render_mode=None throughout), same as
test_rl_policy.py -- D-022's rendering bug doesn't apply at this level.
"""

import gymnasium as gym
import pytest

pytest.importorskip("mani_skill")

import task_schema_draft  # noqa: E402, F401
from atr.envs.tidy_up_policies import _TRAY_SLOTS, attempt_goal, feasibility_aware_policy  # noqa: E402
from atr.language.goal_graph import canonical_example  # noqa: E402
from atr.policies.imitation import (  # noqa: E402
    bc_action,
    collect_demonstrations,
    imitation_policy,
    train_bc_table,
)
from atr.policies.q_learning import ATTEMPT, SKIP  # noqa: E402
from task_schema_draft.rl_policy import learned_policy, train_q_table_canonical  # noqa: E402

_GRAPH = canonical_example()


def _make_env(intervention_kind: str = "bowl_destroyed", onset_step_range: tuple[int, int] = (5, 15)):
    # collect_demonstrations()/train_q_table() both call make_env(intervention_kind,
    # onset_step_range) positionally, matching rl_policy.py's _make_canonical_env.
    return gym.make(
        "TidyUp-v1", num_envs=1, obs_mode="state", render_mode=None,
        sim_backend="physx_cpu", control_mode="pd_ee_delta_pos",
        intervention_kind=intervention_kind, onset_step_range=onset_step_range,
    )


class TestCollectDemonstrations:
    def test_expert_attempts_iff_feasible(self):
        """The recorded expert action must match the same rule
        feasibility_aware_policy hard-codes -- ATTEMPT for a feasible
        goal, SKIP for an infeasible one -- since that rule is
        collect_demonstrations()'s own expert."""
        demos = collect_demonstrations(
            _make_env, _GRAPH, _TRAY_SLOTS, attempt_goal,
            intervention_kinds=("none", "bowl_destroyed"), n_episodes=10, seed=0,
        )
        for (goal_id, feasible), action in demos:
            assert action == (ATTEMPT if feasible else SKIP)


class TestFullCoverageMatchesQLearningAndOracle:
    """When demonstrations cover the same states Q-learning explores
    (both feasible and infeasible, per goal), behavioral cloning should
    match -- the same toy-scale result D-025 already got for Q-learning
    vs. the hand-coded rule, now for a second, differently-trained
    policy."""

    def test_bc_table_matches_the_expert_rule_at_every_seen_key(self):
        demos = collect_demonstrations(
            _make_env, _GRAPH, _TRAY_SLOTS, attempt_goal,
            intervention_kinds=("none", "bowl_destroyed"), n_episodes=40, seed=0,
        )
        bc_table = train_bc_table(demos)
        assert bc_action(bc_table, ("place_mug", True)) == ATTEMPT
        assert bc_action(bc_table, ("place_bowl", True)) == ATTEMPT
        assert bc_action(bc_table, ("place_bowl", False)) == SKIP

    def test_imitation_policy_matches_feasibility_aware_after_bowl_destroyed(self):
        demos = collect_demonstrations(
            _make_env, _GRAPH, _TRAY_SLOTS, attempt_goal,
            intervention_kinds=("none", "bowl_destroyed"), n_episodes=40, seed=0,
        )
        bc_table = train_bc_table(demos)

        results = {}
        for name, run in [
            ("feasibility_aware", feasibility_aware_policy),
            ("imitation", lambda env: imitation_policy(env, bc_table, _GRAPH, attempt_goal, _TRAY_SLOTS)),
        ]:
            env = _make_env(intervention_kind="bowl_destroyed", onset_step_range=(2, 3))
            try:
                env.reset(seed=0)
                results[name] = run(env)
            finally:
                env.close()

        assert results["imitation"]["goals_achieved"] == results["feasibility_aware"]["goals_achieved"]
        assert results["imitation"]["wasted_steps"] == 0


class TestNarrowCoverageFailsToGeneralize:
    """The actual point of this module (see its docstring): a
    demonstration set with narrower coverage than Q-learning's own
    exploration inherits a real gap Q-learning doesn't have.

    Demonstrating only ever with intervention_kind="bowl_destroyed" means
    place_bowl is *always* infeasible by the time it's checked (the
    intervention always fires before goal 2, given onset_step_bounds and
    place_mug's own attempt duration) -- so ("place_bowl", True) is never
    demonstrated at all. Confirmed directly (not assumed): with this
    setup, place_mug is always demonstrated ATTEMPT and place_bowl always
    SKIP, in exactly equal counts, so the global-majority fallback
    (train_bc_table()'s documented behavior for an unseen key) ties and
    breaks toward SKIP by dict insertion order -- not "IL is inherently
    pessimistic," just this scenario's own exact tie. Either way, the
    concrete, checked consequence is the same: an achievable goal gets
    wrongly abandoned."""

    def test_bowl_true_key_is_never_demonstrated(self):
        demos = collect_demonstrations(
            _make_env, _GRAPH, _TRAY_SLOTS, attempt_goal,
            intervention_kinds=("bowl_destroyed",), n_episodes=40, seed=0,
        )
        assert all(not (goal_id == "place_bowl" and feasible) for (goal_id, feasible), _ in demos)
        bc_table = train_bc_table(demos)
        assert ("place_bowl", True) not in bc_table

    def test_narrow_bc_wrongly_skips_the_achievable_bowl_goal_q_learning_gets_right(self):
        narrow_demos = collect_demonstrations(
            _make_env, _GRAPH, _TRAY_SLOTS, attempt_goal,
            intervention_kinds=("bowl_destroyed",), n_episodes=40, seed=0,
        )
        bc_table = train_bc_table(narrow_demos)
        q_table = train_q_table_canonical(n_episodes=120, seed=0)

        results = {}
        for name, run in [
            ("imitation_narrow", lambda env: imitation_policy(env, bc_table, _GRAPH, attempt_goal, _TRAY_SLOTS)),
            ("learned", lambda env: learned_policy(env, q_table)),
        ]:
            env = _make_env(intervention_kind="none")
            try:
                env.reset(seed=0)
                results[name] = run(env)
            finally:
                env.close()

        assert results["learned"]["goals_achieved"] == 2  # Q-learning explored this state; gets it right
        assert results["imitation_narrow"]["goals_achieved"] == 1  # never demonstrated; wrongly skips
        assert results["imitation_narrow"]["per_goal"]["place_bowl"]["skipped"] is True

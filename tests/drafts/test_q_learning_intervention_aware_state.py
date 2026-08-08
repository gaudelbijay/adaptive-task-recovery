"""D-072: closes the loop D-071 opened. D-071 found that `train_q_table()`'s
`(goal_id, feasible)` state key pools across `intervention_kind`, and that
pooling is what made D-070's Q-value an unreliable, small-sample artifact
rather than a genuine discovery -- the true expected value of the pooled
quantity is statistically ambiguous (bootstrap CI straddles zero), while
the quantity conditional on the risky intervention actually being active
is robustly negative. D-071 fixed this for an explicit Monte-Carlo
calibration; this checks whether Q-learning itself, given the same richer
information, also recovers the decisive conditional answer on its own.

`include_intervention_kind=True` (opt-in, defaulted off, `atr.policies.
q_learning`) keys the state on `(goal_id, feasible, intervention_kind)`
instead. Confirmed directly: across 6 independent training seeds, the
richer key converges to a *stable*, large-magnitude negative Q-value for
`(place_bowl, True, "bowl_destroyed")` every time (-0.24 to -2.02, no
longer the noisy -0.03 to -1.66 range the pooled key produced across the
same seeds) and a confident, near-exact +1.0 for
`(place_bowl, True, "none")` -- Q-learning reliably recovers the decisive
conditional answer once the state key stops averaging it away.
"""

import gymnasium as gym
import pytest

pytest.importorskip("mani_skill")

import task_schema_draft  # noqa: E402, F401
from atr.envs.tidy_up_policies import _TRAY_SLOTS, attempt_goal  # noqa: E402
from atr.language.goal_graph import canonical_example  # noqa: E402
from atr.policies.q_learning import ATTEMPT, SKIP, learned_policy, train_q_table  # noqa: E402

_GRAPH = canonical_example()
_WIDE_ONSET_RANGE = (10, 60)


def _make_env(intervention_kind: str = "bowl_destroyed", onset_step_range=(5, 15)):
    return gym.make(
        "TidyUp-v1", num_envs=1, obs_mode="state", render_mode=None,
        sim_backend="physx_cpu", control_mode="pd_ee_delta_pos",
        intervention_kind=intervention_kind, onset_step_range=onset_step_range,
    )


class TestBackwardCompatibility:
    """The default must stay byte-identical to every existing caller's
    behavior -- opt-in only, per this project's zero-behavior-change
    convention for extending an already widely-used interface."""

    def test_default_state_keys_are_still_two_tuples(self):
        q = train_q_table(
            _make_env, _GRAPH, _TRAY_SLOTS, attempt_goal,
            intervention_kinds=("none", "bowl_destroyed"), onset_step_bounds=_WIDE_ONSET_RANGE,
            n_episodes=60, seed=0,
        )
        assert all(len(key) == 2 for key in q)


class TestInterventionAwareStateRecoversTheDecisiveConditionalAnswer:
    @pytest.fixture(scope="class")
    def q_table(self):
        return train_q_table(
            _make_env, _GRAPH, _TRAY_SLOTS, attempt_goal,
            intervention_kinds=("none", "bowl_destroyed"), onset_step_bounds=_WIDE_ONSET_RANGE,
            n_episodes=150, seed=0, include_intervention_kind=True,
        )

    def test_state_keys_are_three_tuples(self, q_table):
        assert all(len(key) == 3 for key in q_table)

    def test_confidently_skips_the_risky_goal_under_real_risk(self, q_table):
        entry = q_table[("place_bowl", True, "bowl_destroyed")]
        assert entry[SKIP] > entry[ATTEMPT]
        # Magnitude varies by training seed (measured -0.24 to -2.02 across
        # 6 seeds, D-072) -- the qualitative, seed-stable claim is the sign
        # and that it's clearly non-trivial, not a specific magnitude.
        assert entry[ATTEMPT] < -0.1

    def test_confidently_attempts_when_genuinely_safe(self, q_table):
        entry = q_table[("place_bowl", True, "none")]
        assert entry[ATTEMPT] > entry[SKIP]
        assert entry[ATTEMPT] > 0.5

    def test_place_mug_unaffected_either_way(self, q_table):
        for intervention_kind in ("none", "bowl_destroyed"):
            entry = q_table[("place_mug", True, intervention_kind)]
            assert entry[ATTEMPT] > entry[SKIP]

    def test_deployed_policy_correctly_adapts_to_the_active_intervention(self, q_table):
        skip_under_risk = 0
        for seed in range(15):
            env = _make_env(intervention_kind="bowl_destroyed", onset_step_range=_WIDE_ONSET_RANGE)
            try:
                env.reset(seed=seed)
                result = learned_policy(env, q_table, _GRAPH, attempt_goal, _TRAY_SLOTS, include_intervention_kind=True)
            finally:
                env.close()
            if result["per_goal"]["place_bowl"].get("skipped", False):
                skip_under_risk += 1
        assert skip_under_risk == 15

        skip_when_safe = 0
        for seed in range(15):
            env = _make_env(intervention_kind="none", onset_step_range=_WIDE_ONSET_RANGE)
            try:
                env.reset(seed=seed)
                result = learned_policy(env, q_table, _GRAPH, attempt_goal, _TRAY_SLOTS, include_intervention_kind=True)
            finally:
                env.close()
            if result["per_goal"]["place_bowl"].get("skipped", False):
                skip_when_safe += 1
        assert skip_when_safe == 0

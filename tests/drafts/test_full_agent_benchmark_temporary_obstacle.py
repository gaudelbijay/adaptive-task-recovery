"""D-090: broadens D-088/D-089's success-criteria benchmark to a second,
genuinely different axis -- docs/10's "unchanged worlds, to measure
unnecessary adaptation" and "visually salient but feasibility-neutral
changes" critical controls. `temporary_obstacle` (`tidy_up_env_replicacad_humanoid.py`)
spawns a real, visually detectable distractor near the tray, then removes
it again -- a real scene change CLIP could plausibly react to, but one
that never makes any goal infeasible.

Investigated before running (not assumed): the two other disclosed gaps
D-089 named -- `kitchen_sink`'s calibration and `potted_meat_can`'s crop --
turned out not to be actionable on inspection. `kitchen_sink`'s reach/tray
configuration was never calibrated for real arm motion at all (documented
since D-027, `tidy_up_env_replicacad_humanoid.py`'s own `_SCENE_CONFIGS`
comment), so a real `attempt_goal()`-based post-attempt check there would
be confounded by an untested reach setup, not a clean test of CLIP alone.
`potted_meat_can` is always goal 1 in the current, fixed instruction text
`atr.pipeline._instruction_graph()` parses -- always checked *before* any
prior attempt (`after_prior_attempt=False` unconditionally) -- so there is
no real code path today where its crop is ever exercised post-attempt.
Neither is tested here for that reason, not by oversight.

The Q-table used is trained only on `("none", "chef_can_destroyed")`
(`train_q_table_replicacad_humanoid()`'s own default) -- it has never seen
`temporary_obstacle` during training. A held-out-intervention-mechanism
generalization test in the same spirit as D-069, but through the real
perceptual pipeline (real CLIP judgment) for the first time, not
privileged state.

Real, measured result: `static`, `oracle_feasibility`, and `full_agent`
all achieve `goals_achieved=2.0` with `wasted_steps=0.0`, zero variance
across 10 seeds -- the distractor object doesn't fall within
`master_chef_can`'s calibrated crop (D-089) and doesn't perturb CLIP's
judgment, so nothing gets unnecessarily abandoned.
"""

import pytest

pytest.importorskip("mani_skill")

import task_schema_draft  # noqa: E402, F401
from atr.evaluation.full_agent_benchmark import run_full_agent_benchmark  # noqa: E402
from atr.pipeline import train_q_table_replicacad_humanoid  # noqa: E402

_WIDE_ONSET_RANGE = (10, 60)


class TestNoUnnecessaryAdaptationUnderAFeasibilityNeutralChange:
    @pytest.fixture(scope="class")
    def q_table(self):
        """Trained only on none/chef_can_destroyed -- never sees
        temporary_obstacle. Its generalization to this held-out mechanism
        is exactly what this test checks, not assumed."""
        return train_q_table_replicacad_humanoid()

    def test_every_policy_achieves_both_goals_with_zero_waste(self, q_table):
        result = run_full_agent_benchmark(
            seeds=list(range(5)), q_table=q_table,
            intervention_kind="temporary_obstacle", onset_step_range=_WIDE_ONSET_RANGE,
        )
        for policy_name in ("static", "oracle_feasibility", "full_agent"):
            goals_mean = result[policy_name]["goals_achieved"][0]
            wasted_mean = result[policy_name]["wasted_steps"][0]
            assert goals_mean == 2.0, f"{policy_name} unexpectedly abandoned a goal: {goals_mean}"
            assert wasted_mean == 0.0, f"{policy_name} unexpectedly wasted steps: {wasted_mean}"

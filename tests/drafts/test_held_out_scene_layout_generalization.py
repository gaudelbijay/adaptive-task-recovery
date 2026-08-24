"""Tests genuine held-out-scene-layout generalization (D-122) -- D-121
built `SCENE_LAYOUT_SPLITS`/`HELD_OUT_SCENE_LAYOUT` (`atr.evaluation.splits`)
but nothing had ever actually trained on the "train" split (`kitchen_cabinet`/
`kitchen_sink`) and evaluated on `"held_out_scene_layout"` (`third_layout`,
D-121). This does: `train_q_table()` trains against a `make_env` that
randomly alternates between the two train-split scene variants each
episode, then the resulting Q-table is evaluated on `third_layout` --
never seen during training.

Real, measured result (verified via a standalone script first, ai-notes/
decisions.md D-122): the learned policy matches oracle feasibility exactly
on the held-out layout, every one of 10 seeds, zero variance -- and `static`
achieves the same recall but wastes 25 steps every time attempting the
infeasible goal blindly. Not a coincidence to be surprised by, same as
D-069's held-out-intervention finding: the learned policy's state is keyed
on `(goal_id, feasible)`, which never encodes *which apartment layout* is
loaded, only whether a goal is currently feasible. This is the first real
confirmation that holds in practice for the scene-layout axis specifically,
not just assumed to transfer from the intervention-axis result.
"""

import random

import gymnasium as gym
import numpy as np
import pytest

pytest.importorskip("mani_skill")

import task_schema_draft  # noqa: E402, F401
from atr.envs.tidy_up_env_replicacad_humanoid import replicacad_humanoid_example  # noqa: E402
from atr.envs.tidy_up_replicacad_humanoid_policies import (  # noqa: E402
    _TRAY_SLOTS,
    attempt_goal,
    feasibility_aware_policy,
    static_policy,
)
from atr.evaluation.splits import HELD_OUT_SCENE_LAYOUT, SCENE_LAYOUT_TRAIN  # noqa: E402
from atr.feasibility.clip_feasibility import visual_object_exists  # noqa: E402
from atr.policies.q_learning import learned_policy, train_q_table  # noqa: E402

_GRAPH = replicacad_humanoid_example()
_TRAIN_VARIANTS = tuple(spec.scene_variant for spec in SCENE_LAYOUT_TRAIN)
_HELD_OUT_VARIANT = HELD_OUT_SCENE_LAYOUT[0].scene_variant


def test_held_out_layout_perception_fails_loudly_until_calibrated():
    """D-123: privileged-state generalization works (D-122), but the
    registered held-out layout has no valid CLIP calibration. Keep that
    boundary explicit instead of silently borrowing another camera's crop."""
    with pytest.raises(ValueError, match="no calibrated visual config"):
        visual_object_exists(
            np.zeros((512, 512, 3), dtype=np.uint8),
            "master_chef_can",
            _HELD_OUT_VARIANT,
        )


def _make_env(intervention_kind: str = "none", onset_step_range: tuple[int, int] = (4, 6), scene_variant: str = "kitchen_cabinet"):
    return gym.make(
        "TidyUp-ReplicaCAD-Humanoid-v1", num_envs=1, obs_mode="state", render_mode=None,
        sim_backend="physx_cpu", control_mode="pd_joint_pos",
        intervention_kind=intervention_kind, onset_step_range=onset_step_range,
        scene_variant=scene_variant,
    )


def _train_make_env_factory(seed: int):
    """A make_env closure with its own RNG that alternates between the two
    train-split scene variants -- train_q_table() itself has no
    scene_variant parameter (it's env-agnostic by design, D-030/D-040), so
    varying the layout has to happen inside the factory it's given."""
    rng = random.Random(seed)

    def make_env(intervention_kind, onset_step_range):
        variant = rng.choice(_TRAIN_VARIANTS)
        return _make_env(intervention_kind, onset_step_range, scene_variant=variant)

    return make_env


class TestLearnedPolicyGeneralizesToHeldOutSceneLayout:
    @pytest.fixture(scope="class")
    def q_table(self):
        return train_q_table(
            _train_make_env_factory(seed=0), _GRAPH, _TRAY_SLOTS, attempt_goal,
            intervention_kinds=("none", "chef_can_destroyed"), onset_step_bounds=(4, 6),
            n_episodes=120, seed=0,
        )

    def test_train_and_held_out_variants_are_disjoint(self):
        """Sanity check on the experimental setup itself -- confirms the
        split is genuinely disjoint (already checked at the registry level,
        test_splits.py::test_train_and_held_out_are_disjoint_variants, but
        cheap to also confirm right where the training actually happens)."""
        assert _HELD_OUT_VARIANT not in _TRAIN_VARIANTS

    def test_q_table_learned_a_decisive_rule(self, q_table):
        """Confirms training actually converged to something meaningful
        (SKIP favored when infeasible, ATTEMPT favored when feasible) before
        trusting its held-out generalization -- an untrained/indifferent
        table would trivially "match" a policy that never disagrees with it."""
        skip_q, attempt_q = q_table[("place_chef_can", False)][0], q_table[("place_chef_can", False)][1]
        assert skip_q > attempt_q  # SKIP favored when master_chef_can is infeasible
        skip_q, attempt_q = q_table[("place_chef_can", True)][0], q_table[("place_chef_can", True)][1]
        assert attempt_q > skip_q  # ATTEMPT favored when feasible

    def test_learned_policy_matches_oracle_on_the_held_out_layout(self, q_table):
        results = {}
        for name, run in [
            ("oracle", feasibility_aware_policy),
            ("learned", lambda env: learned_policy(env, q_table, _GRAPH, attempt_goal, _TRAY_SLOTS)),
        ]:
            env = _make_env(
                intervention_kind="chef_can_destroyed", onset_step_range=(4, 6),
                scene_variant=_HELD_OUT_VARIANT,
            )
            try:
                env.reset(seed=0)
                results[name] = run(env)
            finally:
                env.close()
        assert results["learned"]["goals_achieved"] == results["oracle"]["goals_achieved"]
        assert results["learned"]["wasted_steps"] == 0

    def test_static_baseline_wastes_steps_the_learned_policy_avoids(self, q_table):
        """Same held-out layout/seed, contrasted against the naive baseline
        -- confirms the learned policy's zero-waste result isn't just
        because this scenario happens to have no waste available at all."""
        env = _make_env(
            intervention_kind="chef_can_destroyed", onset_step_range=(4, 6),
            scene_variant=_HELD_OUT_VARIANT,
        )
        try:
            env.reset(seed=0)
            result = static_policy(env)
        finally:
            env.close()
        assert result["goals_achieved"] == 1
        assert result["wasted_steps"] > 0

    def test_learned_policy_matches_oracle_across_multiple_seeds(self, q_table):
        """Single-seed agreement (above) could be a coincidence; repeats the
        comparison across 10 paired seeds on the held-out layout."""
        for seed in range(10):
            env = _make_env(
                intervention_kind="chef_can_destroyed", onset_step_range=(4, 6),
                scene_variant=_HELD_OUT_VARIANT,
            )
            try:
                env.reset(seed=seed)
                oracle_result = feasibility_aware_policy(env)
            finally:
                env.close()

            env = _make_env(
                intervention_kind="chef_can_destroyed", onset_step_range=(4, 6),
                scene_variant=_HELD_OUT_VARIANT,
            )
            try:
                env.reset(seed=seed)
                learned_result = learned_policy(env, q_table, _GRAPH, attempt_goal, _TRAY_SLOTS)
            finally:
                env.close()

            assert learned_result["goals_achieved"] == oracle_result["goals_achieved"], f"seed={seed}"
            assert learned_result["wasted_steps"] == 0, f"seed={seed}"

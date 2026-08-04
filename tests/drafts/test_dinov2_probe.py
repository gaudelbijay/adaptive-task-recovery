"""Tests for dinov2_probe.py -- stage 4 of docs/00-project-overview.md's
build-up order ("swap in a representation learned from unlabeled data").

Slow: each example is a subprocess that boots ManiSkill + SAPIEN fresh (see
dinov2_probe.py's module docstring for why -- D-022, a confirmed upstream
rendering bug, makes that the only reliable way to collect more than ~2
labeled examples). Kept to the minimum example count that still gives
leave-one-out cross-validation something meaningful to hold out.
"""

import pytest

pytest.importorskip("mani_skill")
pytest.importorskip("torch")
pytest.importorskip("sklearn")

from task_schema_draft.dinov2_probe import (  # noqa: E402
    collect_arm_occluded_examples,
    collect_labeled_examples,
    dinov2_embed,
    fit_and_evaluate_probe,
    fit_probe,
    run_end_to_end_episode_dinov2,
)


class TestDinov2Embedding:
    def test_embedding_shape(self):
        import numpy as np

        crop = np.zeros((200, 200, 3), dtype=np.uint8)
        embedding = dinov2_embed(crop)
        assert embedding.shape == (384,)  # ViT-S/14 CLS token width


class TestLinearProbeOnSelfSupervisedFeatures:
    """Same underlying claim as clip_feasibility.py's tests -- does a feasibility
    signal derived from pixels match privileged state? -- but from a
    representation with no language/label supervision at all, evaluated by
    fitting a probe instead of prompting."""

    def test_probe_separates_present_from_absent(self):
        # 6+6, not the 3+3 this started at -- grown per ai-notes/decisions.md
        # D-026 (a live demonstration run at 10+10 also passed, 100% LOO
        # accuracy; kept the test itself smaller to bound runtime, ~75s here
        # vs ~125s at 10+10).
        examples = collect_labeled_examples(
            "master_chef_can", n_present=6, n_absent=6, seed_start=100
        )
        assert sum(label for _, label in examples) == 6  # sanity: labels as expected
        result = fit_and_evaluate_probe(examples)
        assert result["accuracy"] >= 10 / 12, (
            f"leave-one-out accuracy only {result['accuracy']}, "
            f"predictions={result['predictions']} labels={result['labels']}"
        )

    def test_probe_separates_present_from_absent_on_kitchen_sink(self):
        """D-053: closes the gap D-039 flagged explicitly -- this probe had
        never actually been tested against the "kitchen_sink" scene variant
        (D-027) despite collect_labeled_examples() supporting it since that
        same decision. clip_feasibility.py has 2-scene validation; this is
        DINOv2's first. Different seed_start (200, not 100) so its capture
        files never collide with the kitchen_cabinet test above if both run
        concurrently."""
        examples = collect_labeled_examples(
            "master_chef_can", n_present=6, n_absent=6, seed_start=200,
            scene_variant="kitchen_sink",
        )
        assert sum(label for _, label in examples) == 6
        result = fit_and_evaluate_probe(examples)
        assert result["accuracy"] >= 10 / 12, (
            f"leave-one-out accuracy only {result['accuracy']}, "
            f"predictions={result['predictions']} labels={result['labels']}"
        )


class TestLiveDecisionLoopMatchesOracle:
    """D-054/D-055: closes the harder gap D-039 flagged -- DINOv2 wired into
    a real live decision loop, not just probe-fitting tests. Direct
    counterpart to test_pipeline.py's TestFullPipelineMatchesOracle, same
    claim, DINOv2 instead of CLIP. Fits its own probe (not the LOO one
    above) since a live episode needs a probe that can actually predict on
    a new, unseen frame.

    D-054 found this did NOT cleanly match oracle: by the time the second
    goal's frame renders, G1's arm has already moved (reaching for the
    first goal) and, if that reach succeeded, the first object is now
    sitting in the tray too -- an out-of-distribution shift the probe,
    trained only on arm-at-rest captures, got confidently wrong (81%
    "present" on a genuinely destroyed object).

    D-055 closed it for real, not by tuning this test: added
    collect_arm_occluded_examples() (dinov2_probe.py), which captures
    training examples in that same post-first-attempt state (arm moved,
    first object teleported if successful) via a real attempt_goal() call
    inside capture_episode_subprocess.py's new --attempt-object option.
    A reach-only version of this (moving the arm but not also teleporting
    the first object into the tray) was tried first and did NOT reproduce
    the gap -- a probe trained on arm-at-rest data alone judged those
    examples 12/12 correctly. It was only once the capture also replayed
    the *teleport*, matching everything attempt_goal() actually changes in
    the scene, that a probe trained on arm-at-rest data alone reproduced
    D-054's exact 81% confident misjudgment on the new examples -- real
    confirmation the reproduction (and by extension the original finding)
    was faithful, not a different bug. Verified fixed across 5 held-out
    seeds/conditions (ai-notes/decisions.md D-055), each checked in its own
    fresh process -- checking them in one shared process first gave a
    false regression, an artifact of D-022's render-desync budget (~2
    render-producing resets per process), not a real probe failure; caught
    by noticing the two tests below only budget 2 renders total in the
    same pytest session, same as everywhere else in this project."""

    @pytest.fixture(scope="class")
    def q_table(self):
        from atr.pipeline import train_q_table_replicacad_humanoid

        return train_q_table_replicacad_humanoid(n_episodes=30, seed=0)

    @pytest.fixture(scope="class")
    def probe(self):
        rest_examples = collect_labeled_examples(
            "master_chef_can", n_present=6, n_absent=6, seed_start=400,
        )
        # D-055: arm-occluded examples, not just arm-at-rest -- see class
        # docstring. Without these, this fixture reproduces D-054's gap.
        occluded_examples = collect_arm_occluded_examples(n_present=6, n_absent=6, seed_start=500)
        return fit_probe(rest_examples + occluded_examples)

    def _make_env(self, **kwargs):
        import gymnasium as gym

        import task_schema_draft  # noqa: F401  (registers TidyUp-ReplicaCAD-Humanoid-v1)

        return gym.make(
            "TidyUp-ReplicaCAD-Humanoid-v1", num_envs=1, obs_mode="state",
            render_mode="rgb_array", sim_backend="physx_cpu", control_mode="pd_joint_pos", **kwargs,
        )

    def test_intervention_case_matches_oracle(self, q_table, probe):
        """D-054 originally failed here (perceived_feasible incorrectly
        True). D-055's arm-occluded training examples fix it for real --
        if this regresses, re-check whether the `probe` fixture above
        still includes collect_arm_occluded_examples() before assuming
        the underlying representation gap is back."""
        env = self._make_env(intervention_kind="chef_can_destroyed", onset_step_range=(2, 3))
        try:
            env.reset(seed=0)
            result = run_end_to_end_episode_dinov2(env, q_table, probe)
            oracle_exists = dict(env.unwrapped._exists)
        finally:
            env.close()

        assert oracle_exists["master_chef_can"] is False  # genuinely destroyed
        assert oracle_exists["potted_meat_can"] is True

        by_object = {
            "potted_meat_can": result["per_goal"]["place_potted_meat_can"],
            "master_chef_can": result["per_goal"]["place_master_chef_can"],
        }
        assert by_object["potted_meat_can"]["achieved"]
        assert by_object["master_chef_can"]["perceived_feasible"] is False
        assert by_object["master_chef_can"]["skipped"] is True
        assert by_object["master_chef_can"]["achieved"] is False
        assert result["wasted_steps"] == 0

    def test_no_intervention_case_matches_oracle(self, q_table, probe):
        """Companion to test_intervention_case_matches_oracle above -- same
        arm-occluded frame condition, but nothing was destroyed, so the
        correct judgment is perceived_feasible=True this time. Keeping
        both is what makes either one meaningful: a probe that always
        predicts "present" would pass this one for the wrong reason (see
        D-054's original version of this test, which was that biased-but-
        lucky case)."""
        env = self._make_env(intervention_kind="none")
        try:
            env.reset(seed=0)
            result = run_end_to_end_episode_dinov2(env, q_table, probe)
        finally:
            env.close()

        by_object = {
            "potted_meat_can": result["per_goal"]["place_potted_meat_can"],
            "master_chef_can": result["per_goal"]["place_master_chef_can"],
        }
        assert by_object["master_chef_can"]["perceived_feasible"] is True
        assert result["goals_achieved"] == 2
        assert result["wasted_steps"] == 0

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
    """D-054: attempts to close the harder gap D-039 flagged -- DINOv2
    wired into a real live decision loop, not just probe-fitting tests.
    Direct counterpart to test_pipeline.py's TestFullPipelineMatchesOracle,
    same claim, DINOv2 instead of CLIP. Fits its own probe (not the LOO
    one above) since a live episode needs a probe that can actually
    predict on a new, unseen frame.

    Does NOT cleanly match oracle the way the CLIP version does -- see
    test_intervention_case_reveals_a_real_robustness_gap below, which
    locks in a genuine, confirmed finding instead of asserting the
    (false) hoped-for result: by the time the second goal's frame
    renders, G1's arm has already moved (reaching for the first goal),
    an out-of-distribution shift the probe -- trained only on
    arm-at-rest captures -- gets confidently wrong (81% "present" on a
    genuinely destroyed object). CLIP's zero-shot judgment on the exact
    same frame is correct. Real evidence about representation
    robustness, not a bug in this test's wiring."""

    @pytest.fixture(scope="class")
    def q_table(self):
        from atr.pipeline import train_q_table_replicacad_humanoid

        return train_q_table_replicacad_humanoid(n_episodes=30, seed=0)

    @pytest.fixture(scope="class")
    def probe(self):
        examples = collect_labeled_examples(
            "master_chef_can", n_present=6, n_absent=6, seed_start=400,
        )
        return fit_probe(examples)

    def _make_env(self, **kwargs):
        import gymnasium as gym

        import task_schema_draft  # noqa: F401  (registers TidyUp-ReplicaCAD-Humanoid-v1)

        return gym.make(
            "TidyUp-ReplicaCAD-Humanoid-v1", num_envs=1, obs_mode="state",
            render_mode="rgb_array", sim_backend="physx_cpu", control_mode="pd_joint_pos", **kwargs,
        )

    def test_intervention_case_reveals_a_real_robustness_gap(self, q_table, probe):
        """Locks in D-054's actual finding, the same way D-028's
        TestConfirmedUnreachable locks in a confirmed limitation instead
        of silently re-litigating it. This is NOT the desired long-term
        behavior -- if a future fix (e.g. training the probe on frames
        that include the arm mid-reach, not just at-rest captures) makes
        this pass with perceived_feasible=False, that's real progress and
        this test should be updated to expect it. Until then, this
        documents what actually happens: the object is genuinely
        destroyed, but the probe -- seeing an out-of-distribution frame
        with G1's reaching arm in it -- confidently says "present"
        anyway, so the pipeline wastes an attempt instead of skipping."""
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
        # The actual, confirmed finding -- not the hoped-for CLIP-style result:
        assert by_object["master_chef_can"]["perceived_feasible"] is True, (
            "if this now fails, DINOv2's live-loop robustness genuinely improved -- "
            "update this test to assert the correct behavior instead of reverting it"
        )
        assert by_object["master_chef_can"]["skipped"] is False
        assert by_object["master_chef_can"]["achieved"] is False  # attempted, but nothing there
        assert result["wasted_steps"] > 0  # the real cost of the misclassification

    def test_no_intervention_case_passes_but_does_not_demonstrate_robustness(self, q_table, probe):
        """This case passes, but its passing is NOT evidence the probe is
        working correctly -- see test_intervention_case_reveals_a_real_
        robustness_gap above. The probe appears biased toward predicting
        "present" once the arm has moved into frame (confirmed there with
        81% confidence on a genuinely absent object); here, that same
        bias happens to coincide with the true answer, since nothing was
        destroyed. A biased-but-lucky pass and a genuinely correct
        judgment produce identical output -- worth keeping both tests so
        neither is mistaken for the other."""
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

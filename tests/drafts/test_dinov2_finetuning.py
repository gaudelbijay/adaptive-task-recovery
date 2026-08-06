"""Tests for dinov2_probe.py's fine-tuning functions (D-068) -- the
"fine-tuned" half of docs/10's "pretrained frozen and fine-tuned visual
encoders" required baseline, compared directly against
fit_and_evaluate_probe()'s existing frozen-backbone result.

Slow: each LOO fold does a real forward+backward pass through DINOv2's
last transformer block, on top of the subprocess-per-example capture
cost every other dinov2_probe.py test already pays (D-022).
"""

import pytest

pytest.importorskip("mani_skill")
pytest.importorskip("torch")
pytest.importorskip("sklearn")

from task_schema_draft.dinov2_probe import (  # noqa: E402
    collect_arm_occluded_examples,
    collect_labeled_examples,
    dinov2_embed,
    fit_and_evaluate_finetuned,
    fit_and_evaluate_probe,
    fit_finetuned,
    fit_probe,
    predict_finetuned,
)


class TestFitFinetuned:
    def test_training_actually_changes_the_unfrozen_block(self):
        """Rules out the trivial bug (gradients never flowing) before
        trusting anything downstream -- same discipline D-066 used for
        the task-reward-only encoder."""
        import torch

        examples = collect_labeled_examples("master_chef_can", n_present=6, n_absent=6, seed_start=1400)
        model, head, device = fit_finetuned(examples, unfreeze_last_n_blocks=1, epochs=10, lr=1e-5, seed=0)
        # Compare against a freshly-loaded, never-fine-tuned instance of
        # the same last block's weights.
        from task_schema_draft.dinov2_probe import _finetunable_dinov2_model

        fresh_model, _ = _finetunable_dinov2_model()
        fine_tuned_block = dict(model.blocks[-1].named_parameters())
        fresh_block = dict(fresh_model.blocks[-1].named_parameters())
        changed = any(
            not torch.allclose(fine_tuned_block[name], fresh_block[name]) for name in fine_tuned_block
        )
        assert changed


class TestFrozenVsFinetunedOnTheStandardLooSet:
    """At this project's toy LOO scale (12 examples, the same set CLIP/
    DINOv2's frozen probe were both evaluated against), the frozen probe
    already reaches 100% -- there's no headroom for fine-tuning to add
    accuracy. The real question this baseline answers is whether
    fine-tuning *costs* anything (overfitting/catastrophic forgetting on
    ~11 training examples per fold) -- measured, not assumed."""

    def test_finetuned_matches_the_frozen_probes_loo_accuracy(self):
        examples = collect_labeled_examples("master_chef_can", n_present=6, n_absent=6, seed_start=1500)
        frozen = fit_and_evaluate_probe(examples)
        finetuned = fit_and_evaluate_finetuned(
            examples, unfreeze_last_n_blocks=1, epochs=10, lr=1e-5, seed=0,
        )
        assert finetuned["accuracy"] == frozen["accuracy"]


class TestFinetuningInheritsTheSameOodRobustnessGap:
    """D-068's actual, more interesting finding: fine-tuning the backbone
    does NOT provide extra robustness to D-054's out-of-distribution
    shift (G1's arm entering the calibrated crop) for free. Both the
    frozen probe and the fine-tuned encoder, trained on identical
    arm-at-rest-only data (D-054's original, narrow setup, not D-055's
    fix), fail identically on arm-occluded examples they were never
    shown -- reinforcing D-055's own conclusion that the fix is about
    *training data coverage*, not about how much of the model is
    allowed to update. Locked in as a regression test, same pattern as
    D-054's own TestLiveDecisionLoopMatchesOracle before the D-055 fix:
    if this now fails, fine-tuning's robustness genuinely improved on
    its own -- update this test to expect the correct behavior instead
    of reverting it."""

    def test_frozen_probe_reproduces_d054s_gap(self):
        rest_examples = collect_labeled_examples("master_chef_can", n_present=6, n_absent=6, seed_start=1600)
        occluded_examples = collect_arm_occluded_examples(n_present=6, n_absent=6, seed_start=1700)
        probe = fit_probe(rest_examples)  # arm-at-rest only, D-054's original setup
        correct = sum(
            bool(probe.predict(dinov2_embed(crop).reshape(1, -1))[0]) == label
            for crop, label in occluded_examples
        )
        assert correct < len(occluded_examples)  # confirmed gap, not assumed

    def test_finetuned_encoder_does_not_fix_it_either(self):
        rest_examples = collect_labeled_examples("master_chef_can", n_present=6, n_absent=6, seed_start=1800)
        occluded_examples = collect_arm_occluded_examples(n_present=6, n_absent=6, seed_start=1900)
        model, head, device = fit_finetuned(
            rest_examples, unfreeze_last_n_blocks=1, epochs=10, lr=1e-5, seed=0,
        )
        correct = sum(
            predict_finetuned(model, head, device, crop) == label for crop, label in occluded_examples
        )
        assert correct < len(occluded_examples), (
            "if this now passes, fine-tuning the backbone genuinely fixed the OOD gap on its "
            "own -- update this test to expect the correct behavior instead of reverting it"
        )

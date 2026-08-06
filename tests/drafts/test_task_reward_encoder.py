"""Tests for atr.feasibility.task_reward_encoder (D-066) -- the "pixels
trained only through task reward" baseline H1 (docs/01) actually asks
for, previously unbuilt. Slow: leave-one-out evaluation trains a fresh
encoder per fold (12 examples -> 12 short training runs), on top of the
same subprocess-per-example capture cost `dinov2_probe.py`'s own tests
already pay (D-022).
"""

import pytest

pytest.importorskip("mani_skill")
pytest.importorskip("torch")

import task_schema_draft  # noqa: E402, F401
from atr.feasibility.task_reward_encoder import (  # noqa: E402
    fit_and_evaluate_encoder,
    predict_logit,
    train_task_reward_encoder,
)
from task_schema_draft.dinov2_probe import collect_labeled_examples  # noqa: E402


class TestTrainTaskRewardEncoder:
    def test_training_actually_changes_the_weights(self):
        """Rules out the trivial bug (gradients never flowing at all)
        before trusting anything downstream."""
        import torch

        from atr.feasibility.task_reward_encoder import _build_model

        examples = collect_labeled_examples("master_chef_can", n_present=6, n_absent=6, seed_start=800)
        before = _build_model(seed=0)[0].weight.clone()
        model = train_task_reward_encoder(examples, epochs=300, lr=0.01, seed=0)
        after = model[0].weight.clone()
        assert not torch.allclose(before, after)


class TestFromScratchEncoderFailsToDiscriminate:
    """D-066's actual, measured, root-caused finding -- not assumed: a
    from-scratch (no pretrained backbone at all) encoder, given the same
    toy-scale data CLIP and DINOv2 both handle with 100% LOO accuracy,
    fails to learn any real visual discrimination. The mechanism, checked
    directly rather than inferred from the accuracy number alone: the
    model's output is a near-constant, independent of which image is
    fed in -- it converges to predicting whichever fold's own training
    data happens to be the majority class, not anything about the
    image content."""

    def test_predictions_are_near_constant_regardless_of_image_content(self):
        examples = collect_labeled_examples("master_chef_can", n_present=6, n_absent=6, seed_start=900)
        model = train_task_reward_encoder(examples, epochs=300, lr=0.01, seed=0)
        logits = [predict_logit(model, crop) for crop, _ in examples]
        logit_range = max(logits) - min(logits)
        # Real per-example visual signal would produce meaningfully
        # different logits per image; a collapsed model doesn't -- this
        # is the actual signature of the pathology, not the accuracy
        # number below (which is a downstream symptom of it).
        assert logit_range < 0.05, f"logits varied more than expected: {logits}"

    def test_loo_accuracy_is_at_or_below_chance(self):
        """Confirmed 0% in the investigation that motivated this test
        (ai-notes/decisions.md D-066) -- not re-asserting that exact
        number here (a fresh capture could plausibly land on a different
        LOO ordering), but the qualitative claim this baseline exists to
        support: it does not come close to CLIP's/DINOv2's ~100%."""
        examples = collect_labeled_examples("master_chef_can", n_present=6, n_absent=6, seed_start=1000)
        result = fit_and_evaluate_encoder(examples, epochs=300, lr=0.01, seed=0)
        assert result["accuracy"] <= 0.5

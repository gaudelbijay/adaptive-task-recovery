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


class TestFromScratchEncoderFailsToGeneralize:
    """Measure held-out behavior without hard-coding a training mechanism.

    D-066's original captures produced fold-wise majority-class outputs.  A
    fresh Jarvis capture instead fit the balanced training set strongly while
    retaining chance-or-worse leave-one-out accuracy.  Only the held-out
    failure reproduces, so the contract must not require in-sample collapse.
    """

    def test_loo_accuracy_is_at_or_below_chance(self):
        """Require the reproducible result, not D-066's historical mechanism."""
        examples = collect_labeled_examples("master_chef_can", n_present=6, n_absent=6, seed_start=1000)
        result = fit_and_evaluate_encoder(examples, epochs=300, lr=0.01, seed=0)
        assert result["accuracy"] <= 0.5

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

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from aggregate_selected_visual_causal_ood import (  # noqa: E402
    paired_cluster_interval,
    safe_success,
    validate_eval,
)


def test_safe_success_excludes_constraint_violations():
    assert safe_success({"success_once": 1, "constraint_violated": 0}) == 1
    assert safe_success({"success_once": 1, "constraint_violated": 1}) == 0
    assert safe_success({"success_at_end": 0, "constraint_violated": 0}) == 0


def test_paired_cluster_interval_preserves_direction_and_seed_structure():
    groups = [np.ones(32), np.ones(32), np.ones(32)]
    mean, interval = paired_cluster_interval(
        groups, np.random.default_rng(3), repetitions=1000,
    )
    assert mean == 1.0
    assert interval == [1.0, 1.0]
    with pytest.raises(ValueError, match="equal episode counts"):
        paired_cluster_interval(
            [np.ones(2), np.ones(3)], np.random.default_rng(3), 10,
        )


def test_evaluation_validation_fails_closed_on_checkpoint_mismatch():
    payload = {
        "protocol": "held-out deterministic restricted-input visual-policy evaluation",
        "method": "rgb", "training_seed": 7, "condition": "nominal",
        "episodes": 2, "episode_records": [{}, {}], "checkpoint_sha256": "bad",
    }
    with pytest.raises(ValueError, match="checkpoint hash mismatch"):
        validate_eval(
            payload, method="rgb", seed=7, condition="nominal", episodes=2,
            checkpoint_sha256="good",
        )

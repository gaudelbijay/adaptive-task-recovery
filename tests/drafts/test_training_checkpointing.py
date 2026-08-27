import json
import random

import pytest

from atr.training.checkpointing import JsonCheckpointManager, TrainingCheckpoint


def _checkpoint(fingerprint="abc", episodes=10, score=0.5):
    rng = random.Random(7)
    rng.random()
    return TrainingCheckpoint(
        schema_version=1,
        config_fingerprint=fingerprint,
        completed_episodes=episodes,
        learner_state={"q_table": []},
        rng_state_repr=repr(rng.getstate()),
        validation_score=score,
    )


def test_latest_and_best_are_atomic_and_validation_selected(tmp_path):
    manager = JsonCheckpointManager(tmp_path, "abc")
    assert manager.save(_checkpoint(episodes=10, score=0.7))
    assert not manager.save(_checkpoint(episodes=20, score=0.6))

    assert manager.load_latest().completed_episodes == 20
    assert manager.load_best().completed_episodes == 10
    assert not list(tmp_path.glob("*.tmp.*"))
    json.loads((tmp_path / "latest.json").read_text())


def test_rng_state_round_trips_exactly(tmp_path):
    manager = JsonCheckpointManager(tmp_path, "abc")
    checkpoint = _checkpoint()
    manager.save(checkpoint)
    restored = random.Random()
    restored.setstate(manager.load_latest().rng_state())

    expected = random.Random()
    expected.setstate(checkpoint.rng_state())
    assert [restored.random() for _ in range(20)] == [expected.random() for _ in range(20)]


def test_incompatible_configuration_is_rejected(tmp_path):
    JsonCheckpointManager(tmp_path, "old").save(_checkpoint(fingerprint="old"))
    with pytest.raises(ValueError, match="configuration mismatch"):
        JsonCheckpointManager(tmp_path, "new").load_latest()

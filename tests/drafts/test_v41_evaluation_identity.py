import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def load(path):
    return json.loads((ROOT / path).read_text())


def test_evaluation_identity_preserves_the_exact_v40_task():
    evaluation = load("configs/visual_recovery_v19_continuous_canonical_v36.json")
    training = load("configs/visual_recovery_v40_three_seed_view.json")
    assert evaluation["name"] == training["name"]
    assert evaluation["seeds"] == training["seeds"]
    assert evaluation["experiments"] == training["experiments"]

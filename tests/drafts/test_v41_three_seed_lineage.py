import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def load(path): return json.loads((ROOT / path).read_text())


def test_three_seed_views_preserve_exact_tasks():
    pairs = [
        ("configs/visual_recovery_v19_continuous_canonical_v36_smoke.json", "configs/visual_recovery_v36_three_seed_view.json"),
        ("configs/visual_recovery_v19_cardinality_aligned_v38_smoke.json", "configs/visual_recovery_v38_three_seed_view.json"),
        ("configs/visual_recovery_v19_backkey_v40_smoke.json", "configs/visual_recovery_v40_three_seed_view.json"),
    ]
    for smoke_path, view_path in pairs:
        smoke, view = load(smoke_path), load(view_path)
        assert view["name"] == smoke["name"]
        assert view["experiments"] == smoke["experiments"]
        assert view["learning_rate"] == smoke["learning_rate"]
        assert view["weight_decay"] == smoke["weight_decay"]
        assert view["log_freq"] == smoke["log_freq"]
        assert view["seeds"] == [9351, 4796, 1788]


def test_v41_untouched_suite_still_has_three_training_seeds_reserved():
    confirm = load("configs/v36_confirmatory_unseen_visual_ood_v1.json")
    assert confirm["seed_base"] == 117000000
    assert len(confirm["variants"]) == 9

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
load = lambda path: json.loads((ROOT / path).read_text())


def test_v43_is_a_bounded_v42_repair_from_the_same_v40_source():
    v42 = load("configs/visual_recovery_v42_broad_render_smoke.json")
    v43 = load("configs/visual_recovery_v43_identity_bounded_smoke.json")
    left, right = v42["experiments"][0], v43["experiments"][0]
    differing = {key for key in left if left.get(key) != right.get(key)}
    assert differing == {"method", "fine_tune_updates", "total_timesteps", "identity_weight"}
    assert right["identity_weight"] == 5 * left["identity_weight"]
    assert right["fine_tune_updates"] == 1600
    assert v43["learning_rate"] == v42["learning_rate"] / 4
    assert right["source_visual_checkpoint"] == left["source_visual_checkpoint"]


def test_v43_keeps_v42_gate_and_development_protocol_exact():
    v42 = load("configs/v42_smoke_development_ood_v1.json")
    v43 = load("configs/v43_smoke_development_ood_v1.json")
    assert v43["conditions"] == v42["conditions"]
    assert v43["episodes"] == v42["episodes"]
    assert v43["num_envs"] == v42["num_envs"]
    assert v43["seed_base"] == v42["seed_base"]
    assert v43["variants"] == v42["variants"]
    gate42 = load("configs/v42_broad_render_smoke_gate_v1.json")
    gate43 = load("configs/v43_identity_bounded_smoke_gate_v1.json")
    assert gate43["thresholds"] == gate42["thresholds"]

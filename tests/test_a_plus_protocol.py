import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_confirmation_seeds_are_distinct_and_frozen():
    gate = json.loads((ROOT / "configs/a_plus_recovery_gate_v1.json").read_text())
    assert len({gate["development_seed_base"], gate["selection_seed_base"], gate["confirmation_seed_base"]}) == 3
    assert gate["pass_criteria"]["confirmation_must_be_untouched"] is True
    assert gate["readme_release_rule"].startswith("Do not replace")


def test_reboot_snapshot_is_pinned_and_object_disjoint_capable():
    config = json.loads((ROOT / "configs/reboot_external_benchmark_v1.json").read_text())
    rows = config["repositories"]
    assert len(rows) == 37
    assert all(len(row["sha"]) == 40 for row in rows)
    objects = {row["object"] for row in rows}
    assert len(objects) == 9
    for name in objects:
        labels = {row["recovery"] for row in rows if row["object"] == name}
        assert labels == {False, True}


def test_primary_router_data_excludes_evaluator_labels():
    source = (ROOT / "scripts/collect_v4_option_router_data.py").read_text()
    feature_section = source.split("def extract_features", 1)[1].split("def labels", 1)[0]
    assert "critic_intervention_mechanism" not in feature_section
    assert "critic_goal_resolved" not in feature_section
    assert "intervention_target" not in feature_section


def test_router_collection_uses_pre_action_prefixes():
    source = (ROOT / "scripts/collect_v4_option_router_data.py").read_text()
    loop = source.split("for step in range", 1)[1].split("finally:", 1)[0]
    assert loop.index("extract_features(") < loop.index("env.step(action)")
    assert "pre_action_observation_matching_deployment" in source


def test_static_baseline_does_not_receive_hand_engineered_history():
    source = (ROOT / "scripts/collect_v4_option_router_data.py").read_text()
    feature = source.split("def extract_features", 1)[1].split("def labels", 1)[0]
    assert "positions - initial_positions" not in feature
    assert "positions - previous_positions" not in feature
    assert 'qpos"].float() - initial_qpos' not in feature
    assert '"hand_engineered_temporal_features": False' in source


def test_learned_router_has_no_mechanism_state_machine():
    source = (ROOT / "scripts/evaluate_v4_learned_option_router.py").read_text()
    runtime = source.split("def main", 1)[1].split('"forbidden_runtime_inputs"', 1)[0]
    assert "critic_intervention_mechanism" not in runtime
    assert "critic_goal_resolved" not in runtime
    assert "intervention_target" not in runtime
    assert "MOTION_THRESHOLD" not in runtime
    assert "BLOCKAGE_DECISION_HORIZON" not in runtime

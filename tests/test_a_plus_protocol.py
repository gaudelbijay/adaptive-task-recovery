import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_confirmation_seeds_are_distinct_and_frozen():
    gate = json.loads((ROOT / "configs/a_plus_recovery_gate_v1.json").read_text())
    assert len({gate["development_seed_base"], gate["selection_seed_base"], gate["confirmation_seed_base"]}) == 3
    assert gate["pass_criteria"]["confirmation_must_be_untouched"] is True
    assert gate["readme_release_rule"].startswith("Do not replace")


def test_v2_temporal_composition_gate_is_frozen_and_label_disjoint():
    gate = json.loads(
        (ROOT / "configs/a_plus_recovery_gate_v2_temporal_composition.json").read_text()
    )
    assert gate["status"] == "preregistered_before_v2_training_or_evaluation"
    assert gate["representation"]["geometry_dimensions"] == 42
    assert gate["representation"]["heldout_option_index"] == 2
    assert gate["selection_seed_base"] != gate["confirmation_seed_base"]
    assert gate["pass_criteria"]["confirmation_must_be_untouched"] is True
    assert gate["pass_criteria"]["gain_over_strongest_non_oracle_min_pp"] >= 5


def test_v3_centers_the_complete_named_geometry_contract():
    gate = json.loads((ROOT / "configs/a_plus_recovery_gate_v3_full_geometry.json").read_text())
    assert gate["status"] == "preregistered_before_v3_training_or_evaluation"
    assert gate["representation"]["geometry_dimensions"] == 24 + 18 + 15
    assert gate["pass_criteria"] == json.loads(
        (ROOT / "configs/a_plus_recovery_gate_v2_temporal_composition.json").read_text()
    )["pass_criteria"]
    assert gate["selection_seed_base"] != gate["confirmation_seed_base"]


def test_v4_changes_only_the_shared_nominal_controller_and_preserves_gates():
    v3 = json.loads((ROOT / "configs/a_plus_recovery_gate_v3_full_geometry.json").read_text())
    v4 = json.loads((ROOT / "configs/a_plus_recovery_gate_v4_nominal_state.json").read_text())
    assert v4["status"] == "preregistered_before_v4_nominal_training_or_evaluation"
    assert v4["representation"] == v3["representation"]
    assert v4["pass_criteria"] == v3["pass_criteria"]
    assert v4["ood_axes"] == v3["ood_axes"]
    assert v4["shared_option_change"]["unchanged_safe_hold_until_step"] == 36
    assert v4["selection_seed_base"] == 329_000_000
    assert v4["confirmation_seed_base"] == 333_000_000
    assert v4["selection_seed_base"] != v4["confirmation_seed_base"]


def test_external_peg_gate_is_distinct_matched_and_untouched():
    gate = json.loads(
        (ROOT / "configs/a_plus_external_peg_insertion_gate_v1.json").read_text()
    )
    assert gate["status"] == "preregistered_before_external_task_implementation_or_outcomes"
    assert gate["task"]["base_environment"] == "PegInsertionSide-v1"
    assert gate["task"]["native_success_preserved"] is True
    assert "pose assignment is permitted only at reset" in gate["task"]["execution_rule"]
    assert gate["method_contract"]["same_observation_tensor_for_learned_methods"] is True
    assert gate["method_contract"]["same_specialist_checkpoints_for_all_routers"] is True
    assert len({
        gate["development_seed_base"], gate["selection_seed_base"],
        gate["confirmation_seed_base"],
    }) == 3
    assert gate["pass_criteria"]["minimum_independent_training_seeds"] >= 3
    assert gate["pass_criteria"]["confirmation_must_be_untouched"] is True


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


def test_v4_state_nominal_is_shared_by_nominal_and_temporary_options():
    source = (ROOT / "scripts/evaluate_v4_learned_option_router.py").read_text()
    assert 'parser.add_argument(\n        "--nominal-state-checkpoint"' in source
    assert 'state_specs["nominal"]' in source
    assert "temporary_action = nominal_action" in source
    assert '"nominal_policy_type": "state_ppo"' in source
    wrapper = (ROOT / "scripts/slurm_evaluate_v4_learned_option_router.sh").read_text()
    assert "ATR_NOMINAL_STATE_CHECKPOINT" in wrapper

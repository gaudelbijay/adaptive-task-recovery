import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = (ROOT / "src/atr/envs/peg_insertion_recovery.py").read_text()


def test_external_task_inherits_the_official_native_benchmark():
    assert "class PegInsertionRecoveryEnv(PegInsertionSideEnv)" in SOURCE
    assert 'register_env("PegInsertionRecovery-v1"' in SOURCE
    assert "super().evaluate()" in SOURCE


def test_runtime_interventions_use_forces_not_pose_assignment():
    runtime = SOURCE.split("def _before_simulation_step", 1)[1].split(
        "def evaluate", 1
    )[0]
    assert "apply_force" in runtime or "_apply_batched_force" in runtime
    assert "set_pose" not in runtime
    assert "set_linear_velocity" not in runtime


def test_heldout_direction_and_matched_reversible_control_exist():
    assert '"positive_lateral_peg_ejection"' in SOURCE
    assert '"negative_lateral_peg_ejection"' in SOURCE
    assert '"permanent_hole_block"' in SOURCE
    assert '"temporary_hole_block"' in SOURCE
    assert '"critic_physical_unavailable"' in SOURCE


def test_router_geometry_is_physical_and_excludes_mechanism_labels():
    geometry = SOURCE.split('"router_task_geometry"', 1)[1].split("),\n        })", 1)[0]
    for entity in (
        "self.peg.pose.raw_pose",
        "self.box_hole_pose.raw_pose",
        "self.hole_blocker.pose.raw_pose",
        "self.agent.tcp.pose.raw_pose",
    ):
        assert entity in geometry
    assert "_intervention_mechanism" not in geometry
    assert "_physical_unavailable" not in geometry


def test_blocker_contact_is_a_hard_protected_region_violation():
    runtime = SOURCE.split("def _before_simulation_step", 1)[1].split(
        "def evaluate", 1
    )[0]
    assert "peg_head_blocker_distance" in runtime
    assert "blocker_clearance" in runtime
    assert "blocker_protected & (peg_head_blocker_distance < blocker_clearance)" in runtime


def test_fail_fast_smoke_covers_every_frozen_condition():
    smoke = (ROOT / "scripts/smoke_external_peg_recovery.py").read_text()
    for condition in (
        "nominal",
        "positive_lateral_peg_ejection",
        "negative_lateral_peg_ejection",
        "permanent_hole_block",
        "temporary_hole_block",
    ):
        assert f'"{condition}"' in smoke
    assert "ejection_observed_rate" in smoke
    assert "blocker_engaged_rate" in smoke
    assert "temporary_cleared_rate" in smoke


def test_nominal_competence_uses_official_solver_without_intervention():
    audit = (ROOT / "scripts/audit_external_peg_nominal_controller.py").read_text()
    assert "mani_skill.examples.motionplanning.panda.solutions.peg_insertion_side" in audit
    assert 'intervention_probability=0.0' in audit
    assert 'info["success"]' in audit
    assert 'info["constraint_violated"]' in audit
    wrapper = (ROOT / "scripts/slurm_audit_external_peg_nominal_controller.sh").read_text()
    assert "#SBATCH --array=0-31" in wrapper
    assert '"${SLURM_ARRAY_TASK_ID}"' in wrapper


def test_official_ppo_nominal_config_is_pinned_and_three_seed():
    config = json.loads((ROOT / "configs/external_peg_nominal_ppo_v1.json").read_text())
    assert len(config["source_baseline"]["commit"]) == 40
    assert config["seeds"] == [9351, 4796, 1788]
    task = config["experiments"][0]
    assert task["env_id"] == "PegInsertionRecovery-v1"
    assert task["total_timesteps"] == 250_000_000
    assert task["num_envs"] == 1024
    assert task["env_kwargs"]["intervention_probability"] == 0.0


def test_ppo_competence_audit_uses_fresh_development_seeds_and_no_intervention():
    audit = (ROOT / "scripts/evaluate_external_peg_ppo.py").read_text()
    assert 'default=421_000_000' in audit
    assert '"intervention_probability": 0.0' in audit
    assert 'agent.get_action(observation, deterministic=True)' in audit
    assert 'info["success"]' in audit
    assert 'info["constraint_violated"]' in audit
    wrapper = (ROOT / "scripts/slurm_evaluate_external_peg_ppo.sh").read_text()
    assert "#SBATCH --array=0-2" in wrapper
    summary = (ROOT / "scripts/summarize_external_peg_ppo_competence.py").read_text()
    assert 'criteria["minimum_three_seed_mean_safe_success"]' in summary
    assert 'criteria["minimum_per_seed_safe_success"]' in summary
    assert 'criteria["maximum_constraint_violation_rate"]' in summary


def test_external_router_collection_is_causal_group_disjoint_and_heldout():
    collector = (ROOT / "scripts/collect_external_peg_router_data.py").read_text()
    assert '"router_task_geometry"' in collector
    assert 'current_centered_geometry_dim": 12' in collector
    assert '"heldout_option": 2' in collector
    assert '"heldout_option_cross_entropy": False' in collector
    assert '"prefix_timestamp": "pre_action_observation_matching_deployment"' in collector
    assert '"split_unit": "entire vectorized simulator reset batch"' in collector
    feature_section = collector.split("def relative_geometry", 1)[0]
    assert "critic_intervention_mechanism" not in feature_section
    assert "critic_physical_unavailable" not in feature_section


def test_external_closed_loop_evaluator_is_matched_and_scores_abstention():
    evaluator = (ROOT / "scripts/evaluate_external_peg_router.py").read_text()
    for method in (
        "causal_gru", "static_mlp", "unstructured_gru", "heuristic", "oracle",
    ):
        assert f'"{method}"' in evaluator
    assert 'current_centered_sequence(sequence, length, geometry_dim)' in evaluator
    assert 'args.forward_checkpoint or args.nominal_checkpoint' in evaluator
    assert 'args.reverse_checkpoint or args.nominal_checkpoint' in evaluator
    assert 'safe_abstention |= abstained' in evaluator
    assert 'available_success &= info["intervention_finished"].bool()' in evaluator
    assert 'blocker_protected & (peg_head_blocker_distance < blocker_clearance)' in SOURCE
    learned_runtime = evaluator.split("def learned_option", 1)[1].split(
        "def heuristic_option", 1
    )[0]
    assert "critic_intervention_mechanism" not in learned_runtime
    assert "critic_physical_unavailable" not in learned_runtime


def test_external_gate_summarizer_enforces_every_frozen_endpoint():
    summary = (ROOT / "scripts/summarize_external_peg_gate.py").read_text()
    for key in (
        "closed_loop_safe_recovery_min", "native_nominal_success_min",
        "heldout_ejection_safe_recovery_min", "permanent_safe_abstention_min",
        "violation_rate_max", "gain_over_strongest_non_oracle_min_pp",
        "gain_newcombe_95_lower_min_pp", "minimum_independent_training_seeds",
    ):
        assert f'criteria["{key}"]' in summary
    assert 'name not in set(args.oracle)' in summary

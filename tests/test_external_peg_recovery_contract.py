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

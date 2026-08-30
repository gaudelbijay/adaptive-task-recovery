import json
import subprocess
import sys
from pathlib import Path


SCRIPT = Path("scripts/compare_temporal_ssl_continuation.py")
TREATMENT = "temporal"
CONTROL = "no_temporal"


def _episode(success):
    return {
        "success_at_end": float(success), "constraint_violated": 0.0,
        "first_goal_removed": 0.0, "instruction_red_first": 0.0,
    }


def _nominal(method, successes):
    seed_results = []
    for seed in (1, 2, 3):
        records = [_episode(value) for value in successes]
        seed_results.append({
            "training_seed": seed, "batch_seeds": [seed * 10, seed * 10 + 1],
            "episode_records": records,
        })
    return {
        "protocol": "held-out deterministic restricted-input visual-policy evaluation",
        "conditions": {"nominal": {"methods": [{
            "method": method, "episodes": 6, "safe_success_rate": sum(successes) / 2,
            "constraint_violation_rate": 0.0, "seed_results": seed_results,
        }]}},
    }


def _metrics(path, success=1.0):
    evaluation = {
        "success_once": success, "success_at_end": success,
        "constraint_violated": 0.0, "fail_once": 0.0, "fail_at_end": 0.0,
        "goals_completed": 2.0, "goals_unavailable": 0.0,
        "visual_progress_bit_accuracy": 1.0,
    }
    path.write_text(json.dumps({"global_step": 0, "eval": evaluation}) + "\n")


def test_temporal_comparator_confirms_paired_effect_and_fails_on_seed_mismatch(tmp_path):
    strict = tmp_path / "strict.json"
    temporal = tmp_path / "temporal.json"
    control = tmp_path / "control.json"
    config = tmp_path / "config.json"
    output = tmp_path / "effect.json"
    for seed in (1, 2, 3):
        _metrics(tmp_path / f"temporal_{seed}.jsonl")
        _metrics(tmp_path / f"control_{seed}.jsonl")
    effect = {
        "left": "t", "right": "c", "safe_success_rate_difference": 1.0,
        "safe_paired_bootstrap_95": [1.0, 1.0],
        "paired_training_seeds": 3, "paired_episodes": 6,
    }
    branch = {"safe_success_rate": 1.0}
    strict.write_text(json.dumps({
        "protocol": "held-out deterministic strict-actual-removal policy evaluation",
        "cohorts": [
            {"label": "t", "method": TREATMENT, "training_seeds": [1, 2, 3],
             "episodes": 6, "safe_success_rate": 1.0, "constraint_violation_rate": 0.0,
             "first_goal_physically_removed": branch,
             "second_goal_physically_removed": branch},
            {"label": "c", "method": CONTROL, "training_seeds": [1, 2, 3],
             "episodes": 6, "safe_success_rate": 0.0, "constraint_violation_rate": 0.0,
             "first_goal_physically_removed": {"safe_success_rate": 0.0},
             "second_goal_physically_removed": {"safe_success_rate": 0.0}},
        ],
        "paired_comparisons": [effect],
        "paired_comparisons_by_branch": {
            "first_goal_physically_removed": [effect],
            "second_goal_physically_removed": [effect],
        },
    }))
    temporal.write_text(json.dumps(_nominal(TREATMENT, [1, 1])))
    control.write_text(json.dumps(_nominal(CONTROL, [0, 0])))
    config.write_text(json.dumps({
        "treatment": TREATMENT, "control": CONTROL,
        "required_training_seeds": [1, 2, 3],
        "required_episodes_per_condition": 6,
        "treatment_initial_metrics": str(tmp_path / "temporal_{seed}.jsonl"),
        "control_initial_metrics": str(tmp_path / "control_{seed}.jsonl"),
        "strict_aggregate": str(strict),
        "treatment_nominal_aggregate": str(temporal),
        "control_nominal_aggregate": str(control),
        "minimum_worst_endpoint_improvement": 0.03,
        "maximum_treatment_violation_rate": 0.05,
        "confirmation_rule": "test", "claim_boundary": "test",
    }))
    command = [sys.executable, str(SCRIPT), "--config", str(config),
               "--output", str(output)]
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    payload = json.loads(output.read_text())
    assert payload["confirmed"] is True
    assert payload["worst_endpoint_improvement"] == 1.0
    assert payload["paired_initialization_verified"] is True
    assert len(payload["paired_initial_evaluations"]) == 3

    broken = json.loads(control.read_text())
    broken["conditions"]["nominal"]["methods"][0]["seed_results"][0]["batch_seeds"] = [8, 9]
    control.write_text(json.dumps(broken))
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode != 0
    assert "paired reset seeds mismatch" in result.stderr

    control.write_text(json.dumps(_nominal(CONTROL, [0, 0])))
    _metrics(tmp_path / "control_2.jsonl", success=0.0)
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode != 0
    assert "step-zero paired initialization mismatch for seed 2" in result.stderr

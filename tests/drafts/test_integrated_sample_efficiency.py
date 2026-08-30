import json
import subprocess
import sys
from pathlib import Path


SCRIPT = Path("scripts/build_integrated_sample_efficiency.py")


def test_sample_accounting_joins_sources_and_fails_on_bad_arithmetic(tmp_path):
    performance = tmp_path / "performance.json"
    contract = tmp_path / "contract.json"
    output = tmp_path / "paper" / "sample"
    performance.write_text(json.dumps({
        "protocol": "matched strict-recovery and nominal-retention comparison",
        "required_training_seeds": 3,
        "cohorts": [{
            "label": "visual", "method": "m",
            "strict_safe_success_rate": 0.92,
            "nominal_safe_success_rate": 0.94,
            "worst_case_safe_success_rate": 0.90,
            "strict_violation_rate": 0.01,
            "nominal_violation_rate": 0.02,
        }],
    }))
    method = {
        "label": "M", "method": "m", "modality": "visual",
        "deployed_actor_inputs": "RGB + proprioception",
        "training_only_asymmetric_critic": True,
        "training_only_state_teacher": True,
        "strict_state_teacher_checkpoint": "teacher.pt",
        "training_only_goal_resolution_labels": False,
        "training_seeds": [1, 2, 3],
        "executed_ppo_interactions_per_seed": 100,
        "dagger_interactions_per_seed": 20,
        "new_interactions_per_seed": 120,
        "new_interactions_all_seeds": 360,
        "reported_interactions_exclude_upstream_training": True,
    }
    contract.write_text(json.dumps({
        "protocol": "configuration-derived method information and interaction accounting",
        "methods": [method],
    }))
    command = [
        sys.executable, str(SCRIPT), "--performance", str(performance),
        "--method-contract", str(contract), "--output-prefix", str(output),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    payload = json.loads(output.with_suffix(".json").read_text())
    assert payload["rows"][0]["new_interactions_per_seed"] == 120
    assert "92.00%" in output.with_suffix(".md").read_text()
    assert output.with_suffix(".csv").stat().st_size > 0

    method["new_interactions_all_seeds"] = 359
    contract.write_text(json.dumps({
        "protocol": "configuration-derived method information and interaction accounting",
        "methods": [method],
    }))
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode != 0
    assert "all-seed interaction arithmetic mismatch" in result.stderr

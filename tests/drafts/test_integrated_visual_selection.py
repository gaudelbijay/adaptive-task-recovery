import json
import subprocess
import sys
from pathlib import Path


SCRIPT = Path("scripts/select_integrated_visual_policy.py")
STRICT_PROTOCOL = "held-out deterministic strict-actual-removal policy evaluation"
NOMINAL_PROTOCOL = "held-out deterministic restricted-input visual-policy evaluation"
SEMANTICS = "event_reward_intervention_target_only_v3"


def write_inputs(tmp_path, strict_safe=0.93, nominal_safe=0.92, first=0.90,
                 second=0.96, strict_violation=0.01, nominal_violation=0.02):
    method = "candidate"
    strict_path = tmp_path / "strict.json"
    nominal_path = tmp_path / "nominal.json"
    config_path = tmp_path / "config.json"
    output = tmp_path / "selected.json"
    strict_path.write_text(json.dumps({
        "protocol": STRICT_PROTOCOL, "benchmark_semantics": SEMANTICS,
        "cohorts": [{
            "label": "candidate", "method": method, "episodes": 768,
            "training_seeds": [1, 2, 3], "safe_success_rate": strict_safe,
            "constraint_violation_rate": strict_violation,
            "first_goal_physically_removed": {"safe_success_rate": first},
            "second_goal_physically_removed": {"safe_success_rate": second},
        }],
    }))
    nominal_path.write_text(json.dumps({
        "protocol": NOMINAL_PROTOCOL, "benchmark_semantics": SEMANTICS,
        "conditions": {"nominal": {"methods": [{
            "method": method, "episodes": 768, "seeds": 3,
            "safe_success_rate": nominal_safe,
            "constraint_violation_rate": nominal_violation,
        }]}},
    }))
    config_path.write_text(json.dumps({
        "name": "selection", "strict_aggregate": str(strict_path),
        "candidates": [{"label": "candidate", "method": method,
                        "nominal_aggregate": str(nominal_path)}],
        "thresholds": {
            "minimum_strict_safe": 0.90, "minimum_nominal_safe": 0.90,
            "minimum_first_removed_safe": 0.85,
            "minimum_second_removed_safe": 0.85,
            "maximum_strict_violation": 0.05,
            "maximum_nominal_violation": 0.05,
        },
        "claim_boundary": "test",
    }))
    return config_path, output


def run(tmp_path, **kwargs):
    config, output = write_inputs(tmp_path, **kwargs)
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--config", str(config), "--output", str(output)],
        text=True, capture_output=True,
    )
    return result, json.loads(output.read_text()) if output.exists() else None


def test_integrated_selection_accepts_only_balanced_safe_candidate(tmp_path):
    result, payload = run(tmp_path)
    assert result.returncode == 0
    assert payload["selected"] == "candidate"
    assert payload["candidates"][0]["worst_case_safe_success_rate"] == 0.90
    assert all(payload["candidates"][0]["checks"].values())


def test_integrated_selection_rejects_branch_collapse(tmp_path):
    result, payload = run(tmp_path, first=0.84)
    assert result.returncode == 0
    assert payload["selected"] is None
    assert payload["all_candidates_ineligible"] is True
    assert payload["candidates"][0]["checks"]["first_removed_safe"] is False


def test_integrated_selection_rejects_nominal_or_safety_regression(tmp_path):
    result, payload = run(tmp_path, nominal_safe=0.89, strict_violation=0.06)
    assert result.returncode == 0
    assert payload["selected"] is None
    assert payload["candidates"][0]["checks"]["nominal_safe"] is False
    assert payload["candidates"][0]["checks"]["strict_violation"] is False


def test_integrated_selection_fails_on_incomplete_cohort(tmp_path):
    config, output = write_inputs(tmp_path)
    strict_path = Path(json.loads(config.read_text())["strict_aggregate"])
    strict = json.loads(strict_path.read_text())
    strict["cohorts"][0]["episodes"] = 767
    strict_path.write_text(json.dumps(strict))
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--config", str(config), "--output", str(output)],
        text=True, capture_output=True,
    )
    assert result.returncode != 0
    assert not output.exists()

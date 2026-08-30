import json
import subprocess
import sys
from pathlib import Path


SCRIPT = Path("scripts/build_integrated_regime_comparison.py")
SEMANTICS = "event_reward_intervention_target_only_v3"


def test_integrated_regime_builder_fails_closed_and_writes_all_formats(tmp_path):
    strict = tmp_path / "strict.json"; nominal = tmp_path / "nominal.json"
    state_nominal = tmp_path / "state_nominal.json"
    config = tmp_path / "config.json"; output = tmp_path / "paper" / "integrated"
    branch = {"safe_success_rate": 0.91,
              "safe_success_hierarchical_bootstrap_95": [0.88, 0.94]}
    strict.write_text(json.dumps({
        "protocol": "held-out deterministic strict-actual-removal policy evaluation",
        "benchmark_semantics": SEMANTICS,
        "cohorts": [{"label": "visual", "kind": "visual", "method": "m",
                     "episodes": 768, "training_seeds": [1, 2, 3],
                     "safe_success_rate": 0.92,
                     "safe_success_hierarchical_bootstrap_95": [0.89, 0.95],
                     "constraint_violation_rate": 0.01,
                     "first_goal_physically_removed": branch,
                     "second_goal_physically_removed": branch},
                    {"label": "state", "kind": "state", "method": "s",
                     "episodes": 768, "training_seeds": [1, 2, 3],
                     "safe_success_rate": 0.95,
                     "safe_success_hierarchical_bootstrap_95": [0.93, 0.97],
                     "constraint_violation_rate": 0.0,
                     "first_goal_physically_removed": branch,
                     "second_goal_physically_removed": branch}],
    }))
    nominal.write_text(json.dumps({
        "protocol": "held-out deterministic restricted-input visual-policy evaluation",
        "benchmark_semantics": SEMANTICS,
        "conditions": {"nominal": {"methods": [{"method": "m", "episodes": 768,
            "seeds": 3, "safe_success_rate": 0.94,
            "safe_success_hierarchical_bootstrap_95": [0.92, 0.96],
            "constraint_violation_rate": 0.02}]}},
    }))
    state_nominal.write_text(json.dumps({
        "protocol": "held-out deterministic state-policy evaluation",
        "benchmark_semantics": SEMANTICS,
        "nominal_condition": [{"method": "s", "episodes": 768, "seeds": 3,
            "pooled_safe_success_rate": 0.93,
            "safe_success_hierarchical_bootstrap_95": [0.90, 0.95],
            "constraint_violation_rate": 0.01}],
    }))
    config.write_text(json.dumps({"name": "x", "strict_aggregate": str(strict),
        "cohorts": [{"label": "visual", "kind": "visual", "method": "m",
                     "nominal_aggregate": str(nominal)},
                    {"label": "state", "kind": "state", "method": "s",
                     "nominal_aggregate": str(state_nominal)}],
        "claim_boundary": "test"}))
    result = subprocess.run([sys.executable, str(SCRIPT), "--config", str(config),
                             "--output-prefix", str(output)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    payload = json.loads(output.with_suffix(".json").read_text())
    assert payload["cohorts"][0]["worst_case_safe_success_rate"] == 0.91
    assert payload["cohorts"][1]["nominal_safe_success_rate"] == 0.93
    assert "92.00% [89.00, 95.00]" in output.with_suffix(".md").read_text()
    for suffix in (".json", ".md", ".csv", ".png", ".pdf"):
        assert output.with_suffix(suffix).stat().st_size > 0
    expanded_strict = json.loads(strict.read_text())
    for cohort in expanded_strict["cohorts"]:
        cohort["episodes"] = 1280
        cohort["training_seeds"] = [1, 2, 3, 4, 5]
    strict.write_text(json.dumps(expanded_strict))
    for path, section in ((nominal, "conditions"), (state_nominal, "nominal_condition")):
        data = json.loads(path.read_text())
        record = (data[section]["nominal"]["methods"][0]
                  if section == "conditions" else data[section][0])
        record["episodes"] = 1280
        record["seeds"] = 5
        path.write_text(json.dumps(data))
    expanded_config = json.loads(config.read_text())
    expanded_config["required_episodes"] = 1280
    expanded_config["required_training_seeds"] = 5
    config.write_text(json.dumps(expanded_config))
    result = subprocess.run([sys.executable, str(SCRIPT), "--config", str(config),
                             "--output-prefix", str(output)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert json.loads(output.with_suffix(".json").read_text())[
        "required_training_seeds"
    ] == 5
    broken = json.loads(strict.read_text()); broken["cohorts"][0]["episodes"] = 767
    strict.write_text(json.dumps(broken))
    result = subprocess.run([sys.executable, str(SCRIPT), "--config", str(config),
                             "--output-prefix", str(output)], capture_output=True, text=True)
    assert result.returncode != 0
    broken["cohorts"][0]["episodes"] = 1280
    broken["cohorts"][0]["safe_success_hierarchical_bootstrap_95"] = [0.96, 0.89]
    strict.write_text(json.dumps(broken))
    result = subprocess.run([sys.executable, str(SCRIPT), "--config", str(config),
                             "--output-prefix", str(output)], capture_output=True, text=True)
    assert result.returncode != 0

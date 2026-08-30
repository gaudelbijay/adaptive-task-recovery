import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from check_dual_specialist_teacher_gate import check  # noqa: E402


def write(path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")


def config(tmp_path):
    return {
        "nominal_aggregate": str(tmp_path / "nominal.json"),
        "strict_aggregate": str(tmp_path / "strict.json"),
        "nominal_method": "nominal",
        "strict_label": "strict_label",
        "strict_method": "strict",
        "seeds": [1, 2, 3],
        "episodes_per_condition": 768,
        "thresholds": {
            "minimum_nominal_raw": 0.90,
            "minimum_nominal_safe": 0.90,
            "minimum_strict_safe": 0.90,
            "minimum_first_removed_safe": 0.85,
            "minimum_second_removed_safe": 0.85,
            "maximum_nominal_violation": 0.05,
            "maximum_strict_violation": 0.05,
        },
        "claim_boundary": "test",
    }


def artifacts(tmp_path, nominal_safe=0.96):
    write(tmp_path / "nominal.json", {
        "protocol": "held-out deterministic restricted-input visual-policy evaluation",
        "benchmark_semantics": "event_reward_intervention_target_only_v3",
        "conditions": {"nominal": {"methods": [{
            "method": "nominal", "episodes": 768, "seeds": 3,
            "success_rate": 0.97, "safe_success_rate": nominal_safe,
            "constraint_violation_rate": 0.01,
        }]}},
    })
    write(tmp_path / "strict.json", {
        "protocol": "held-out deterministic strict-actual-removal policy evaluation",
        "benchmark_semantics": "event_reward_intervention_target_only_v3",
        "cohorts": [{
            "label": "strict_label", "method": "strict", "episodes": 768,
            "training_seeds": [1, 2, 3], "safe_success_rate": 0.98,
            "constraint_violation_rate": 0.0,
            "first_goal_physically_removed": {"safe_success_rate": 0.97},
            "second_goal_physically_removed": {"safe_success_rate": 0.98},
        }],
    })


def test_dual_specialist_gate_passes_only_regime_qualified_teachers(tmp_path):
    artifacts(tmp_path)
    payload = check(config(tmp_path))
    assert payload["passed"] is True
    assert all(payload["checks"].values())


def test_dual_specialist_gate_fails_closed_on_weak_nominal_teacher(tmp_path):
    artifacts(tmp_path, nominal_safe=0.89)
    payload = check(config(tmp_path))
    assert payload["passed"] is False
    assert payload["checks"]["nominal_safe"] is False

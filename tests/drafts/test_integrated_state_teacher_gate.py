import copy
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from check_integrated_state_teacher_gate import check  # noqa: E402


def inputs(tmp_path):
    method = "teacher"
    strict_path = tmp_path / "strict.json"
    nominal_path = tmp_path / "nominal.json"
    strict_path.write_text(json.dumps({
        "protocol": "held-out deterministic strict-actual-removal policy evaluation",
        "benchmark_semantics": "event_reward_intervention_target_only_v3",
        "cohorts": [{
            "label": "teacher", "method": method,
            "training_seeds": [9351, 4796, 1788], "episodes": 768,
            "safe_success_rate": 0.94, "constraint_violation_rate": 0.01,
            "first_goal_physically_removed": {"safe_success_rate": 0.92},
            "second_goal_physically_removed": {"safe_success_rate": 0.96},
        }],
    }))
    nominal_path.write_text(json.dumps({
        "protocol": "held-out deterministic state-policy evaluation",
        "benchmark_semantics": "event_reward_intervention_target_only_v3",
        "experiment": "state_experiment",
        "environments": [{"method": method, "seeds": 3, "episodes": 768}],
        "nominal_condition": [{
            "method": method, "seeds": 3, "episodes": 768,
            "pooled_safe_success_rate": 0.93,
            "constraint_violation_rate": 0.02,
        }],
    }))
    return {
        "strict_aggregate": str(strict_path),
        "state_aggregate": str(nominal_path),
        "expected_experiment": "state_experiment",
        "strict_label": "teacher", "method": method,
        "seeds": [9351, 4796, 1788], "episodes_per_condition": 768,
        "thresholds": {
            "minimum_strict_safe": 0.90, "minimum_nominal_safe": 0.90,
            "minimum_first_removed_safe": 0.85,
            "minimum_second_removed_safe": 0.85,
            "maximum_strict_violation": 0.05,
            "maximum_nominal_violation": 0.05,
        },
        "claim_boundary": "test",
    }


def test_integrated_teacher_gate_accepts_only_complete_dual_regime_teacher(tmp_path):
    config = inputs(tmp_path)
    result = check(config)
    assert result["passed"] is True
    assert all(result["checks"].values())
    assert result["training_seeds"] == [9351, 4796, 1788]
    assert all(len(value) == 64 for value in result["source_sha256"].values())

    failed = copy.deepcopy(config)
    nominal_path = Path(failed["state_aggregate"])
    nominal = json.loads(nominal_path.read_text())
    nominal["nominal_condition"][0]["pooled_safe_success_rate"] = 0.89
    nominal_path.write_text(json.dumps(nominal))
    result = check(failed)
    assert result["passed"] is False
    assert result["checks"]["nominal_safe"] is False


def test_integrated_teacher_gate_rejects_seed_or_episode_drift(tmp_path):
    config = inputs(tmp_path)
    strict_path = Path(config["strict_aggregate"])
    strict = json.loads(strict_path.read_text())
    strict["cohorts"][0]["training_seeds"] = [9351, 4796, 9999]
    strict_path.write_text(json.dumps(strict))
    try:
        check(config)
    except ValueError as error:
        assert "wrong training seeds" in str(error)
    else:
        raise AssertionError("seed drift passed the teacher gate")

    config = inputs(tmp_path)
    nominal_path = Path(config["state_aggregate"])
    nominal = json.loads(nominal_path.read_text())
    nominal["nominal_condition"][0]["episodes"] = 767
    nominal_path.write_text(json.dumps(nominal))
    try:
        check(config)
    except ValueError as error:
        assert "wrong episode count" in str(error)
    else:
        raise AssertionError("incomplete nominal evaluation passed the teacher gate")


def test_production_gate_uses_state_only_strict_artifact():
    gate = json.loads(Path(
        "configs/integrated_state_teacher_gate_v1.json"
    ).read_text())
    strict = json.loads(Path(
        "configs/strict_removal_integrated_state_teacher_gate_v1.json"
    ).read_text())
    assert gate["strict_aggregate"].endswith(
        "strict_removal_integrated_state_teacher_gate_v1/aggregate.json"
    )
    assert len(strict["cohorts"]) == 1
    assert strict["cohorts"][0]["label"] == "integrated_mixture_state"

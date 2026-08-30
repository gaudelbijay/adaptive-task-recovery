import json
import subprocess
import sys
from pathlib import Path


SCRIPT = Path("scripts/check_state_teacher_nominal_gate.py")


def run_gate(tmp_path, record):
    aggregate = tmp_path / "aggregate.json"
    output = tmp_path / "gate.json"
    aggregate.write_text(json.dumps({"nominal_condition": [record]}))
    result = subprocess.run(
        [
            sys.executable, str(SCRIPT), "--aggregate", str(aggregate),
            "--method", "teacher", "--output", str(output),
        ],
        text=True, capture_output=True,
    )
    return result, json.loads(output.read_text()) if output.exists() else None


def test_state_teacher_gate_passes_only_complete_competent_cohort(tmp_path):
    result, payload = run_gate(tmp_path, {
        "method": "teacher", "episodes": 768, "seeds": 3,
        "pooled_success_rate": 0.91, "pooled_safe_success_rate": 0.89,
        "constraint_violation_rate": 0.01,
    })
    assert result.returncode == 0
    assert payload["passed"] is True
    assert payload["checks"] == {"raw": True, "safe": True, "violation": True}
    assert len(payload["aggregate_sha256"]) == 64


def test_state_teacher_gate_rejects_unsafe_or_weak_teacher(tmp_path):
    result, payload = run_gate(tmp_path, {
        "method": "teacher", "episodes": 768, "seeds": 3,
        "pooled_success_rate": 0.90, "pooled_safe_success_rate": 0.69,
        "constraint_violation_rate": 0.06,
    })
    assert result.returncode == 1
    assert payload["passed"] is False
    assert payload["checks"] == {"raw": True, "safe": False, "violation": False}


def test_state_teacher_gate_rejects_incomplete_cohort(tmp_path):
    result, payload = run_gate(tmp_path, {
        "method": "teacher", "episodes": 512, "seeds": 2,
        "pooled_success_rate": 1.0, "pooled_safe_success_rate": 1.0,
        "constraint_violation_rate": 0.0,
    })
    assert result.returncode != 0
    assert payload is None

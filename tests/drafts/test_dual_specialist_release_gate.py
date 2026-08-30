import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from check_dual_specialist_release_gate import check  # noqa: E402


PROTOCOL = "predeclared integrated state-teacher allocation gate"


def setup(tmp_path, primary=False, fallback=False):
    primary_path = tmp_path / "primary.json"
    fallback_path = tmp_path / "fallback.json"
    primary_path.write_text(json.dumps({
        "protocol": PROTOCOL, "method": "primary", "passed": primary,
    }), encoding="utf-8")
    fallback_path.write_text(json.dumps({
        "protocol": PROTOCOL, "method": "fallback", "passed": fallback,
    }), encoding="utf-8")
    return {
        "primary_gate": str(primary_path),
        "fallback_gate": str(fallback_path),
        "expected_protocol": PROTOCOL,
        "primary_method": "primary",
        "fallback_method": "fallback",
        "claim_boundary": "test",
    }


def test_release_requires_two_actual_failed_gate_artifacts(tmp_path):
    payload = check(setup(tmp_path))
    assert payload["eligible"] is True
    assert all(payload["checks"].values())


def test_release_rejects_either_passing_integrated_teacher(tmp_path):
    assert check(setup(tmp_path, primary=True))["eligible"] is False
    assert check(setup(tmp_path, fallback=True))["eligible"] is False


def test_release_fails_closed_when_fallback_never_ran(tmp_path):
    config = setup(tmp_path)
    Path(config["fallback_gate"]).unlink()
    with pytest.raises(FileNotFoundError):
        check(config)

import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from check_state_fallback_release_gate import check  # noqa: E402


PROTOCOL = "predeclared integrated state-teacher allocation gate"


def config(tmp_path, passed):
    path = tmp_path / "primary.json"
    path.write_text(json.dumps({
        "protocol": PROTOCOL, "method": "primary", "passed": passed,
    }), encoding="utf-8")
    return {
        "primary_gate": str(path),
        "expected_protocol": PROTOCOL,
        "primary_method": "primary",
        "claim_boundary": "test",
    }


def test_state_fallback_requires_explicit_primary_failure(tmp_path):
    payload = check(config(tmp_path, False))
    assert payload["eligible"] is True
    assert payload["checks"]["primary_gate_ran_and_failed"] is True


def test_state_fallback_rejects_passing_primary(tmp_path):
    assert check(config(tmp_path, True))["eligible"] is False


def test_state_fallback_fails_closed_without_primary_artifact(tmp_path):
    payload = config(tmp_path, False)
    Path(payload["primary_gate"]).unlink()
    with pytest.raises(FileNotFoundError):
        check(payload)

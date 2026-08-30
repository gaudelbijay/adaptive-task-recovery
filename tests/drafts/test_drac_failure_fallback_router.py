import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

import route_drac_failure_fallback as router  # noqa: E402


def test_router_preserves_explicit_gate_verdict(monkeypatch, tmp_path):
    config = tmp_path / "gate.json"
    config.write_text(json.dumps({"sentinel": 1}))
    monkeypatch.setattr(router, "check", lambda _: {"eligible": True})
    result = router.resolve(config)
    assert result["eligible"] is True
    assert result["resolution"] == "eligible"
    assert result["config_sha256"]


def test_router_releases_failure_path_on_ineligible_gate(monkeypatch, tmp_path):
    config = tmp_path / "gate.json"
    config.write_text(json.dumps({"sentinel": 1}))
    monkeypatch.setattr(router, "check", lambda _: {"eligible": False})
    result = router.resolve(config)
    assert result["eligible"] is False
    assert result["resolution"] == "ineligible"


def test_router_fails_closed_and_records_gate_error(monkeypatch, tmp_path):
    config = tmp_path / "gate.json"
    config.write_text(json.dumps({"sentinel": 1}))

    def fail(_):
        raise FileNotFoundError("missing completion sentinel")

    monkeypatch.setattr(router, "check", fail)
    result = router.resolve(config)
    assert result["eligible"] is False
    assert result["resolution"] == "gate_error"
    assert result["error_type"] == "FileNotFoundError"
    assert "missing completion sentinel" in result["error"]
    assert "performance claim" in result["claim_boundary"]

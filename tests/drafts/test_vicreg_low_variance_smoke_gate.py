import copy
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from check_vicreg_low_variance_smoke_gate import check  # noqa: E402


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n")


def fixture(tmp_path: Path) -> dict:
    task = {
        "method": "baseline", "total_timesteps": 100_000,
        "temporal_variance_coefficient": 0.01, "num_envs": 10,
        "num_steps": 10, "shared": True,
    }
    baseline = {
        "name": "base", "seeds": [9351, 1, 2], "experiments": [task],
        "selection_failure_penalty": 2.0, "claim_boundary": "base",
    }
    candidate = copy.deepcopy(baseline)
    candidate.update(name="candidate", seeds=[9351], claim_boundary="smoke")
    candidate["experiments"][0].update(
        method="candidate", total_timesteps=1_000,
        temporal_variance_coefficient=0.001,
    )
    extension = copy.deepcopy(baseline)
    extension.update(name="extension", claim_boundary="extension")
    extension["experiments"][0].update(
        method="extension", temporal_variance_coefficient=0.001,
    )
    base_config = tmp_path / "base.json"
    candidate_config = tmp_path / "candidate.json"
    extension_config = tmp_path / "extension.json"
    write_json(base_config, baseline); write_json(candidate_config, candidate)
    write_json(extension_config, extension)
    root = tmp_path / "results"
    base_dir = root / "base" / "baseline" / "seed_9351"
    cand_dir = root / "candidate" / "candidate" / "seed_9351"
    base_dir.mkdir(parents=True); cand_dir.mkdir(parents=True)
    base_eval = {"global_step": 1_000, "eval": {
        "success_at_end": 0.65, "constraint_violated": 0.05, "return": 0.0,
    }}
    cand_eval = {"global_step": 1_000, "eval": {
        "success_at_end": 0.90, "constraint_violated": 0.02, "return": 0.0,
    }}
    (base_dir / "metrics.jsonl").write_text(json.dumps(base_eval) + "\n")
    (cand_dir / "metrics.jsonl").write_text(json.dumps(cand_eval) + "\n")
    write_json(cand_dir / "TRAINING_COMPLETE.json", {"global_step": 1_000})
    return {
        "baseline_config": str(base_config),
        "candidate_config": str(candidate_config),
        "extension_config": str(extension_config),
        "results_root": str(root), "seed": 9351,
        "thresholds": {
            "minimum_success_at_end": 0.85,
            "maximum_constraint_violation": 0.05,
            "minimum_safety_weighted_improvement": 0.15,
        },
        "claim_boundary": "allocation only",
    }


def test_gate_passes_only_on_paired_matched_budget_improvement(tmp_path):
    payload = check(fixture(tmp_path))
    assert payload["eligible"] is True
    assert payload["scheduled_step"] == 1_000
    assert payload["safety_weighted_improvement"] == pytest.approx(0.31)


def test_gate_fails_closed_on_unmatched_method_change(tmp_path):
    config = fixture(tmp_path)
    path = Path(config["candidate_config"])
    candidate = json.loads(path.read_text())
    candidate["experiments"][0]["shared"] = False
    write_json(path, candidate)
    with pytest.raises(ValueError, match="outside frozen smoke fields"):
        check(config)


def test_gate_rejects_incomplete_exact_budget(tmp_path):
    config = fixture(tmp_path)
    completion = (
        Path(config["results_root"]) / "candidate" / "candidate" /
        "seed_9351" / "TRAINING_COMPLETE.json"
    )
    write_json(completion, {"global_step": 900})
    with pytest.raises(ValueError, match="exact diagnostic budget"):
        check(config)


def test_gate_fails_closed_on_unmatched_extension_change(tmp_path):
    config = fixture(tmp_path)
    path = Path(config["extension_config"])
    extension = json.loads(path.read_text())
    extension["experiments"][0]["shared"] = False
    write_json(path, extension)
    with pytest.raises(ValueError, match="outside frozen extension fields"):
        check(config)


def test_gate_reports_failed_threshold_without_redefining_it(tmp_path):
    config = fixture(tmp_path)
    metrics = (
        Path(config["results_root"]) / "candidate" / "candidate" /
        "seed_9351" / "metrics.jsonl"
    )
    metrics.write_text(json.dumps({"global_step": 1_000, "eval": {
        "success_at_end": 0.84, "constraint_violated": 0.0, "return": 0.0,
    }}) + "\n")
    payload = check(config)
    assert payload["eligible"] is False
    assert payload["checks"]["success_at_end"] is False

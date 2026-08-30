import copy
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from check_drac_stability_smoke_gate import check, normalized  # noqa: E402


def config(name, method, seeds, timesteps, pad, coefficient=None):
    task = {
        "method": method, "env_id": "LearnedRecovery-v3",
        "total_timesteps": timesteps, "num_envs": 2, "num_steps": 2,
        "augmentation_pad": pad,
    }
    if coefficient is not None:
        task["drac_policy_coefficient"] = coefficient
    return {
        "name": name, "seeds": seeds, "experiments": [task],
        "selection_failure_penalty": 2.0,
    }


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def evaluation(step, end, violation, reward=100):
    return {"global_step": step, "eval": {
        "success_at_end": end, "constraint_violated": violation,
        "return": reward,
    }}


def fixture(tmp_path):
    baseline = config("base", "base_method", [3, 2, 1], 100, 0)
    candidate = config("smoke", "smoke_method", [1], 20, 4, 0.1)
    extension = config("full", "full_method", [3, 2, 1], 100, 4, 0.1)
    paths = [tmp_path / name for name in ("base.json", "smoke.json", "full.json")]
    for path, value in zip(paths, (baseline, candidate, extension)):
        write(path, value)
    root = tmp_path / "results"
    base_dir = root / "base" / "base_method" / "seed_1"
    smoke_dir = root / "smoke" / "smoke_method" / "seed_1"
    base_records = [
        evaluation(8, 0.90, 0.08), evaluation(12, 0.90, 0.08),
        evaluation(16, 0.90, 0.08), evaluation(20, 0.90, 0.08),
    ]
    smoke_records = [
        evaluation(8, 0.91, 0.03), evaluation(12, 0.92, 0.02),
        evaluation(16, 0.93, 0.02), evaluation(20, 0.94, 0.01),
    ]
    for directory, records in ((base_dir, base_records), (smoke_dir, smoke_records)):
        directory.mkdir(parents=True)
        (directory / "metrics.jsonl").write_text(
            "\n".join(json.dumps(item) for item in records) + "\n"
        )
    write(smoke_dir / "TRAINING_COMPLETE.json", {"global_step": 20})
    gate = {
        "baseline_config": str(paths[0]), "candidate_config": str(paths[1]),
        "extension_config": str(paths[2]), "results_root": str(root),
        "seed": 1, "tail_evaluations": 3,
        "thresholds": {
            "minimum_best_success_at_end": 0.9,
            "maximum_best_constraint_violation": 0.05,
            "minimum_best_score_margin": -0.05,
            "maximum_tail_mean_constraint_violation": 0.05,
            "minimum_tail_mean_score_improvement": 0.05,
        }, "claim_boundary": "allocation only",
    }
    return gate, paths, smoke_dir


def test_frozen_repository_configs_are_matched_treatments():
    baseline = json.loads((ROOT / "configs/visual_recovery_dual_specialist_dagger_v19.json").read_text())
    smoke = json.loads((ROOT / "configs/visual_recovery_dual_specialist_drac_v22_smoke.json").read_text())
    full = json.loads((ROOT / "configs/visual_recovery_dual_specialist_drac_v22.json").read_text())
    assert normalized(baseline, remove_budget=True) == normalized(smoke, remove_budget=True)
    assert normalized(baseline, remove_budget=False) == normalized(full, remove_budget=False)


def test_gate_passes_only_complete_safe_and_stable_candidate(tmp_path):
    gate, _, _ = fixture(tmp_path)
    result = check(gate)
    assert result["eligible"] is True
    assert all(result["checks"].values())


def test_gate_fails_closed_on_tail_instability(tmp_path):
    gate, _, smoke_dir = fixture(tmp_path)
    records = [evaluation(8, 0.95, 0.01), evaluation(12, 0.95, 0.10), evaluation(16, 0.95, 0.10), evaluation(20, 0.95, 0.10)]
    (smoke_dir / "metrics.jsonl").write_text(
        "\n".join(json.dumps(item) for item in records) + "\n"
    )
    result = check(gate)
    assert result["eligible"] is False
    assert result["checks"]["tail_mean_constraint_violation"] is False


def test_gate_rejects_unmatched_non_treatment_change(tmp_path):
    gate, paths, _ = fixture(tmp_path)
    candidate = json.loads(paths[1].read_text())
    candidate["experiments"][0]["num_steps"] = 4
    write(paths[1], candidate)
    with pytest.raises(ValueError, match="outside frozen"):
        check(gate)


def test_gate_rejects_inexact_completion(tmp_path):
    gate, _, smoke_dir = fixture(tmp_path)
    write(smoke_dir / "TRAINING_COMPLETE.json", {"global_step": 16})
    with pytest.raises(ValueError, match="exact smoke budget"):
        check(gate)

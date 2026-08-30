import copy
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from check_shift_action_runtime_gate import check, normalized  # noqa: E402


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def fixture(tmp_path):
    runtime_ref = {
        "name": "ref_runtime", "seeds": [1],
        "experiments": [{"method": "ref", "total_timesteps": 8,
                         "num_envs": 2, "num_steps": 2}],
    }
    runtime = copy.deepcopy(runtime_ref)
    runtime["name"] = "runtime"; runtime["experiments"][0]["method"] = "candidate"
    smoke_ref = copy.deepcopy(runtime_ref); smoke_ref["name"] = "ref_smoke"
    smoke = copy.deepcopy(smoke_ref)
    smoke["name"] = "smoke"; smoke["experiments"][0]["method"] = "candidate_smoke"
    paths = [tmp_path / f"{name}.json" for name in ("rr", "r", "sr", "s")]
    for path, value in zip(paths, (runtime_ref, runtime, smoke_ref, smoke)):
        write(path, value)
    run = tmp_path / "results/runtime/candidate/seed_1"
    write(run / "TRAINING_COMPLETE.json", {"global_step": 8})
    records = [
        {"global_step": 0, "eval": {"success_at_end": 0.9,
                                     "constraint_violated": 0.01}},
        {"global_step": 4, "train_loss": {"drac_policy": 0.2}},
        {"global_step": 8, "eval": {"success_at_end": 0.85,
                                     "constraint_violated": 0.02}},
    ]
    (run / "metrics.jsonl").write_text("\n".join(map(json.dumps, records)) + "\n")
    gate = {
        "reference_runtime_config": str(paths[0]),
        "candidate_runtime_config": str(paths[1]),
        "reference_smoke_config": str(paths[2]),
        "candidate_smoke_config": str(paths[3]),
        "results_root": str(tmp_path / "results"), "seed": 1,
        "thresholds": {
            "minimum_final_success_at_end": 0.8,
            "maximum_final_constraint_violation": 0.05,
            "maximum_success_drop_from_initial": 0.15,
            "maximum_logged_consistency_loss": 1.95,
        }, "claim_boundary": "allocation only",
    }
    return gate, paths, run


def test_repository_configs_match_their_frozen_drac_protocol_controls():
    def load(name): return json.loads((ROOT / "configs" / name).read_text())
    assert normalized(load("visual_recovery_dual_specialist_drac_v22_runtime_smoke.json")) == normalized(load("visual_recovery_dual_specialist_shift_action_v24_runtime_smoke.json"))
    assert normalized(load("visual_recovery_dual_specialist_drac_v22_smoke.json")) == normalized(load("visual_recovery_dual_specialist_shift_action_v24_smoke.json"))


def test_gate_passes_complete_bounded_and_retained_runtime(tmp_path):
    gate, _, _ = fixture(tmp_path)
    result = check(gate)
    assert result["eligible"] is True
    assert all(result["checks"].values())


def test_gate_rejects_catastrophic_success_drop(tmp_path):
    gate, _, run = fixture(tmp_path)
    records = [
        {"global_step": 0, "eval": {"success_at_end": 0.9, "constraint_violated": 0}},
        {"global_step": 4, "train_loss": {"drac_policy": 0.2}},
        {"global_step": 8, "eval": {"success_at_end": 0.0, "constraint_violated": 0}},
    ]
    (run / "metrics.jsonl").write_text("\n".join(map(json.dumps, records)) + "\n")
    result = check(gate)
    assert result["eligible"] is False
    assert result["checks"]["final_success"] is False
    assert result["checks"]["success_retention"] is False


def test_gate_rejects_unbounded_or_nonfinite_loss(tmp_path):
    gate, _, run = fixture(tmp_path)
    records = [
        {"global_step": 0, "eval": {"success_at_end": 0.9, "constraint_violated": 0}},
        {"global_step": 4, "train_loss": {"drac_policy": 2.0}},
        {"global_step": 8, "eval": {"success_at_end": 0.9, "constraint_violated": 0}},
    ]
    (run / "metrics.jsonl").write_text("\n".join(map(json.dumps, records)) + "\n")
    result = check(gate)
    assert result["eligible"] is False
    assert result["checks"]["bounded_consistency"] is False


def test_gate_fails_closed_on_config_or_budget_drift(tmp_path):
    gate, paths, run = fixture(tmp_path)
    candidate = json.loads(paths[1].read_text())
    candidate["experiments"][0]["num_steps"] = 4
    write(paths[1], candidate)
    with pytest.raises(ValueError, match="outside method identity"):
        check(gate)
    write(paths[1], json.loads(paths[0].read_text()) | {"name": "runtime"})
    candidate = json.loads(paths[1].read_text())
    candidate["experiments"][0]["method"] = "candidate"; write(paths[1], candidate)
    write(run / "TRAINING_COMPLETE.json", {"global_step": 4})
    with pytest.raises(ValueError, match="exact budget"):
        check(gate)

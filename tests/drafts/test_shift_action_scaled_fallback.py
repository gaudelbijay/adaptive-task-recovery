import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from check_drac_stability_smoke_gate import normalized  # noqa: E402
import route_shift_action_scaled_fallback as router  # noqa: E402
import check_shift_action_scaled_stability_smoke_gate as scaled_gate  # noqa: E402


def load_config(name):
    return json.loads((ROOT / "configs" / name).read_text())


def gate_fixture(config_path, *, eligible):
    checks = {key: True for key in router.EXPECTED_CHECKS}
    if not eligible:
        checks["tail_mean_score_improvement"] = False
    return {
        "protocol": router.PROTOCOL,
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "scheduled_step": router.EXPECTED_SCHEDULED_STEP,
        "checks": checks,
        "eligible": eligible,
        "candidate_training_source_sha256": {"trainer": "a" * 64},
        "candidate_best_checkpoint_sha256": "b" * 64,
    }


def test_v25_configs_are_one_scaled_matched_treatment():
    baseline = load_config("visual_recovery_dual_specialist_dagger_v19.json")
    smoke = load_config(
        "visual_recovery_dual_specialist_shift_action_v25_scaled_smoke.json"
    )
    full = load_config(
        "visual_recovery_dual_specialist_shift_action_v25_scaled.json"
    )
    assert normalized(baseline, remove_budget=True) == normalized(
        smoke, remove_budget=True
    )
    assert normalized(baseline, remove_budget=False) == normalized(
        full, remove_budget=False
    )
    assert smoke["seeds"] == [1788]
    assert full["seeds"] == baseline["seeds"]
    for task in (smoke["experiments"][0], full["experiments"][0]):
        assert task["augmentation_pad"] == 4
        assert task["drac_policy_coefficient"] == 0.02


def test_v25_gate_retains_v24_thresholds_and_mechanical_scale():
    scaled = load_config("shift_action_scaled_stability_smoke_gate_v1.json")
    original = load_config("shift_action_stability_smoke_gate_v1.json")
    assert scaled["thresholds"] == original["thresholds"]
    assert scaled["tail_evaluations"] == original["tail_evaluations"] == 3
    assert scaled["expected_drac_policy_coefficient"] == 0.02
    coefficient_cap = 0.25 * (
        abs(0.008008738954231376) + 0.5 * 0.02259076078189537
    ) / 0.20910160499624908
    assert 0.02 <= coefficient_cap < 0.021 + 0.003


def test_v25_failure_path_replaces_only_unallocated_v24_slot():
    strict_v24 = load_config(
        "strict_removal_dual_specialist_shift_action_extension_v12.json"
    )
    strict_v25 = load_config(
        "strict_removal_dual_specialist_shift_action_scaled_extension_v13.json"
    )
    select_v24 = load_config("integrated_visual_selection_v9.json")
    select_v25 = load_config("integrated_visual_selection_v10.json")
    assert strict_v25["cohorts"][:-1] == strict_v24["cohorts"][:-1]
    assert select_v25["candidates"][:-1] == select_v24["candidates"][:-1]
    assert select_v25["thresholds"] == select_v24["thresholds"]
    assert strict_v25["cohorts"][-1]["config"].endswith(
        "visual_recovery_dual_specialist_shift_action_v25_scaled.json"
    )
    assert select_v25["candidates"][-1]["method"] == (
        "event_reward_dual_specialist_shift_action_scaled_visual_ppo"
    )
    assert "before any V25 metric" in strict_v25["claim_boundary"]
    assert "before any V25 metric" in select_v25["claim_boundary"]
    assert "nonexistent V24" in strict_v25["claim_boundary"]
    assert "nonexistent V24" in select_v25["claim_boundary"]


def test_v25_causal_ood_protocol_changes_only_selection_and_policy_map():
    v1 = load_config("selected_visual_causal_ood_v1.json")
    v2 = load_config("selected_visual_causal_ood_v2.json")
    for key in (
        "conditions", "episodes", "num_envs", "seed_base",
        "hypothesis_thresholds", "variants",
    ):
        assert v2[key] == v1[key]
    assert v2["policy_configs"] | v1["policy_configs"] == v2["policy_configs"]
    assert v2["policy_configs"][
        "scaled_bounded_shift_action_dual_specialist_visual"
    ].endswith("visual_recovery_dual_specialist_shift_action_v25_scaled.json")
    assert v2["selection"].endswith("integrated_visual_selection_v10.json")
    assert "before any V25 metric" in v2["claim_boundary"]


def test_scaled_checker_validates_real_matched_configs_and_sources():
    baseline = load_config("visual_recovery_dual_specialist_dagger_v19.json")
    smoke = load_config(
        "visual_recovery_dual_specialist_shift_action_v25_scaled_smoke.json"
    )
    full = load_config(
        "visual_recovery_dual_specialist_shift_action_v25_scaled.json"
    )
    tasks = scaled_gate.validate_loaded_configs(
        baseline, smoke, full, seed=1788, expected_coefficient=0.02
    )
    assert tasks[1]["total_timesteps"] == 20_000_000
    assert set(scaled_gate.SOURCE_PATHS) == {
        "trainer", "trainer_wrapper", "base_trainer", "environment",
        "environment_v3", "bounded_shift_action_consistency",
    }
    assert all(path.is_file() for path in scaled_gate.SOURCE_PATHS.values())


def test_scaled_checker_rejects_any_coefficient_drift():
    import copy

    baseline = load_config("visual_recovery_dual_specialist_dagger_v19.json")
    smoke = load_config(
        "visual_recovery_dual_specialist_shift_action_v25_scaled_smoke.json"
    )
    full = load_config(
        "visual_recovery_dual_specialist_shift_action_v25_scaled.json"
    )
    changed = copy.deepcopy(smoke)
    changed["experiments"][0]["drac_policy_coefficient"] = 0.02001
    import pytest
    with pytest.raises(ValueError, match="wrong mechanically scaled coefficient"):
        scaled_gate.validate_loaded_configs(
            baseline, changed, full, seed=1788, expected_coefficient=0.02
        )


def test_scaled_checker_end_to_end_eligibility(monkeypatch, tmp_path):
    baseline = load_config("visual_recovery_dual_specialist_dagger_v19.json")
    smoke = load_config(
        "visual_recovery_dual_specialist_shift_action_v25_scaled_smoke.json"
    )
    full = load_config(
        "visual_recovery_dual_specialist_shift_action_v25_scaled.json"
    )
    config = load_config("shift_action_scaled_stability_smoke_gate_v1.json")
    config.update({
        "baseline_config": "baseline.json",
        "candidate_config": "candidate.json",
        "extension_config": "extension.json",
        "results_root": str(tmp_path),
    })
    source_paths = {}
    for name in scaled_gate.SOURCE_PATHS:
        path = tmp_path / f"{name}.py"
        path.write_text(name)
        source_paths[name] = path
    monkeypatch.setattr(scaled_gate, "SOURCE_PATHS", source_paths)
    expected_source = {
        name: hashlib.sha256(path.read_bytes()).hexdigest()
        for name, path in source_paths.items()
    }

    candidate_task = smoke["experiments"][0]
    candidate_dir = (
        tmp_path / smoke["name"] / candidate_task["method"] / "seed_1788"
    )
    candidate_dir.mkdir(parents=True)
    (candidate_dir / "best.pt").write_bytes(b"checkpoint")

    def fake_load(path):
        key = str(path)
        if key == "baseline.json":
            return baseline, "baseline-hash"
        if key == "candidate.json":
            return smoke, "candidate-hash"
        if key == "extension.json":
            return full, "extension-hash"
        if key.endswith("TRAINING_COMPLETE.json"):
            return {"global_step": 19_996_672}, "complete-hash"
        raise AssertionError(key)

    def record(step, success, violation):
        return {
            "global_step": step,
            "eval": {
                "success_at_end": success,
                "constraint_violated": violation,
                "return": 100.0,
            },
        }

    baseline_records = [
        record(18_000_000, 0.84, 0.03),
        record(19_000_000, 0.85, 0.03),
        record(19_996_672, 0.86, 0.03),
    ]
    candidate_records = [
        record(18_000_000, 0.93, 0.01),
        record(19_000_000, 0.94, 0.01),
        record(19_996_672, 0.95, 0.01),
    ]

    monkeypatch.setattr(scaled_gate, "load", fake_load)
    monkeypatch.setattr(
        scaled_gate,
        "evaluations",
        lambda path, _maximum: (
            baseline_records if baseline["name"] in str(path) else candidate_records
        ),
    )
    monkeypatch.setattr(
        scaled_gate.torch,
        "load",
        lambda *_args, **_kwargs: {"source_sha256": expected_source},
    )
    monkeypatch.setattr(
        scaled_gate,
        "bounded_loss_check",
        lambda *_args: {
            "finite_bounded_consistency": True,
            "maximum_logged_consistency_loss": 0.2,
            "consistency_records": 10,
        },
    )
    payload = scaled_gate.check(config)
    assert payload["eligible"] is True
    assert all(payload["checks"].values())
    assert payload["scheduled_step"] == 19_996_672
    assert payload["candidate_best_checkpoint_sha256"] == hashlib.sha256(
        b"checkpoint"
    ).hexdigest()


def test_router_authorizes_only_explicit_rejection(tmp_path):
    config = tmp_path / "gate_config.json"
    config.write_text("{}\n")
    result = tmp_path / "gate_result.json"
    result.write_text(json.dumps(gate_fixture(config, eligible=False)))
    payload = router.resolve(result, config)
    assert payload["authorize_v25"] is True
    assert payload["resolution"] == "v24_explicitly_rejected"
    assert payload["upstream_result_sha256"]


def test_router_suppresses_v25_when_v24_passes(tmp_path):
    config = tmp_path / "gate_config.json"
    config.write_text("{}\n")
    result = tmp_path / "gate_result.json"
    result.write_text(json.dumps(gate_fixture(config, eligible=True)))
    payload = router.resolve(result, config)
    assert payload["authorize_v25"] is False
    assert payload["resolution"] == "v24_eligible"


def test_router_fails_closed_on_missing_or_malformed_evidence(tmp_path):
    config = tmp_path / "gate_config.json"
    config.write_text("{}\n")
    missing = router.resolve(tmp_path / "missing.json", config)
    assert missing["authorize_v25"] is False
    assert missing["resolution"] == "invalid_or_missing_v24_gate"

    result = tmp_path / "gate_result.json"
    malformed = gate_fixture(config, eligible=False)
    malformed["candidate_best_checkpoint_sha256"] = "short"
    result.write_text(json.dumps(malformed))
    rejected = router.resolve(result, config)
    assert rejected["authorize_v25"] is False
    assert rejected["resolution"] == "invalid_or_missing_v24_gate"


def test_router_rejects_inconsistent_eligibility(tmp_path):
    config = tmp_path / "gate_config.json"
    config.write_text("{}\n")
    result = tmp_path / "gate_result.json"
    payload = gate_fixture(config, eligible=False)
    payload["checks"] = {key: True for key in router.EXPECTED_CHECKS}
    result.write_text(json.dumps(payload))
    routed = router.resolve(result, config)
    assert routed["authorize_v25"] is False
    assert "disagrees" in routed["error"]


def test_slurm_router_has_no_gpu_and_uses_after_artifact_contract():
    wrapper = (ROOT / "scripts/slurm_route_shift_action_scaled_fallback.sh").read_text()
    assert "--partition=compute-short" in wrapper
    assert "--gres=" not in wrapper
    assert "shift_action_stability_smoke_gate_v1.json" in wrapper
    assert "route_shift_action_scaled_fallback.py" in wrapper
    gate_wrapper = (
        ROOT / "scripts/slurm_check_shift_action_scaled_stability_smoke_gate.sh"
    ).read_text()
    assert "--partition=compute-short" in gate_wrapper
    assert "--gres=" not in gate_wrapper
    assert "check_shift_action_scaled_stability_smoke_gate.py" in gate_wrapper


def test_submission_dag_routes_before_sbatch_and_keeps_every_gate():
    script = (ROOT / "scripts/submit_shift_action_scaled_fallback.sh").read_text()
    assert script.index("route_shift_action_scaled_fallback.py") < script.index(
        "smoke_job=$(sbatch"
    )
    assert "refusing duplicate V25 submission" in script
    assert "--array=0-2%3" in script
    assert "afterok:${gate_job}" in script
    assert "afterok:${full_job}_*" in script
    assert "slurm_audit_training_checkpoints.sh" in script
    assert "slurm_visual_strict_removal_eval.sh" in script
    assert "ATR_EVAL_CONDITIONS=nominal" in script
    assert "strict_removal_dual_specialist_shift_action_scaled_extension_v13.json" in script
    assert "integrated_visual_selection_v10.json" in script
    assert "--array=0-29%6" in script
    assert "selected_visual_causal_ood_v2.json" in script
    assert "afterok:1140386" in script
    assert "afterok:1140387" in script

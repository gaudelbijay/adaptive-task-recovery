import json
import subprocess
import sys
from pathlib import Path


SCRIPT = Path("scripts/check_integrated_five_seed_gate.py")


def load(path):
    return json.loads(Path(path).read_text())


def test_confirmatory_tasks_are_identical_and_gate_fails_closed(tmp_path):
    pairs = [
        ("configs/visual_recovery_progress_dagger_v6_event_reward.json",
         "configs/visual_recovery_progress_dagger_v6_confirm_append.json",
         "configs/visual_recovery_progress_dagger_v6_five_seed_view.json", True),
        ("configs/visual_recovery_strict_adaptive_v13_stable.json",
         "configs/visual_recovery_strict_adaptive_v13_confirm_append.json",
         "configs/visual_recovery_strict_adaptive_v13_five_seed_view.json", True),
        ("configs/learned_recovery_ppo_v12_integrated_mixture.json",
         "configs/learned_recovery_ppo_v12_integrated_confirm_append.json",
         "configs/learned_recovery_ppo_v12_integrated_five_seed_view.json", True),
        ("configs/learned_recovery_ppo_v11_strict_removal.json",
         "configs/learned_recovery_ppo_v11_strict_removal_confirm_append.json",
         None, False),
        ("configs/visual_recovery_dual_specialist_dagger_v19.json",
         "configs/visual_recovery_dual_specialist_dagger_v19_confirm_append.json",
         None, False),
    ]
    for screen_path, append_path, view_path, check_view in pairs:
        screen, append = load(screen_path), load(append_path)
        assert append["name"] == screen["name"]
        assert append["experiments"] == screen["experiments"]
        assert append["seeds"] == [71064, 84293]
        if check_view:
            view = load(view_path)
            assert view["name"] == screen["name"]
            assert view["experiments"] == screen["experiments"]
            assert view["seeds"] == [9351, 4796, 1788, 71064, 84293]

    selection = tmp_path / "selection.json"
    expected_checks = [
        "strict_safe", "nominal_safe", "first_removed_safe",
        "second_removed_safe", "strict_violation", "nominal_violation",
    ]
    manifest = load("configs/integrated_five_seed_confirmation_v2.json")
    selection.write_text(json.dumps({
        "protocol": "predeclared integrated visual-policy selection",
        "selected": "strict_stable_visual", "all_candidates_ineligible": False,
        "thresholds": manifest["expected_thresholds"],
        "candidates": [{"label": "strict_stable_visual", "eligible": True,
                         "checks": {key: True for key in expected_checks}}],
    }))
    manifest["selection_artifact"] = str(selection)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    output = tmp_path / "gate.json"
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--config", str(manifest_path),
         "--output", str(output)], capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert load(output)["authorized"] is True

    rejected = load(selection)
    rejected["selected"] = None
    rejected["all_candidates_ineligible"] = True
    selection.write_text(json.dumps(rejected))
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--config", str(manifest_path),
         "--output", str(output)], capture_output=True, text=True,
    )
    assert result.returncode != 0


def test_dual_specialist_gate_hashes_every_same_seed_training_config(tmp_path):
    manifest = load("configs/dual_specialist_five_seed_confirmation_v1.json")
    selection = tmp_path / "selection.json"
    selection.write_text(json.dumps({
        "protocol": "predeclared integrated visual-policy selection",
        "selected": "dual_specialist_visual", "all_candidates_ineligible": False,
        "thresholds": manifest["expected_thresholds"],
        "candidates": [{
            "label": "dual_specialist_visual", "eligible": True,
            "checks": {key: True for key in manifest["expected_checks"]},
        }],
    }))
    manifest["selection_artifact"] = str(selection)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    output = tmp_path / "gate.json"
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--config", str(manifest_path),
         "--output", str(output)], capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    payload = load(output)
    assert payload["authorized"] is True
    assert set(payload["training_config_sha256"]) == set(manifest["training_configs"])

    bad = load(manifest["training_configs"][-1])
    bad["seeds"] = [71064]
    bad_path = tmp_path / "bad.json"
    bad_path.write_text(json.dumps(bad))
    manifest["training_configs"][-1] = str(bad_path)
    manifest_path.write_text(json.dumps(manifest))
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--config", str(manifest_path),
         "--output", str(output)], capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert "wrong untouched seeds" in result.stderr

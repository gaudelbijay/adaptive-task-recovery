import json
import subprocess
import sys
from pathlib import Path


SCRIPT = Path("scripts/check_v28_final_release_gate.py")


def write(path, payload):
    path.write_text(json.dumps(payload) + "\n")


def fixtures(tmp_path):
    method = "v19_rendered_domain_distillation"
    standard = {"conditions": {}}
    for condition, safe in (("nominal", 0.91), ("intervention", 0.94)):
        standard["conditions"][condition] = {"methods": [{
            "method": method,
            "safe_success_rate": safe,
            "seed_results": [{"safe_success_rate": safe}] * 3,
        }]}
    strict = {"cohorts": [
        {"label": "v19_visual_incumbent", "safe_success_rate": 0.96,
         "seed_safe_success_rates": [0.95, 0.96, 0.97]},
        {"label": "v28_render_distilled_visual", "safe_success_rate": 0.93,
         "seed_safe_success_rates": [0.91, 0.93, 0.95]},
    ]}
    records = []
    for name in ("camera_right_3cm", "lighting_cool"):
        records.append({
            "variant": name, "variant_safe_success_rate": 0.84,
            "per_seed": [{"variant_safe_success_rate": 0.81}] * 3,
        })
    unseen = {
        "records": records,
        "hypotheses": {
            "selected_policy_is_robust_to_frozen_visual_ood_suite": True,
            "learned_progress_head_has_causal_control_utility": True,
        },
    }
    paths = {name: tmp_path / f"{name}.json" for name in ("standard", "strict", "unseen")}
    for name, payload in (("standard", standard), ("strict", strict), ("unseen", unseen)):
        write(paths[name], payload)
    gate = {
        "schema_version": 1, "name": "test", "candidate_method": method,
        "candidate_strict_label": "v28_render_distilled_visual",
        "incumbent_strict_label": "v19_visual_incumbent",
        "standard_aggregate": str(paths["standard"]),
        "strict_aggregate": str(paths["strict"]),
        "unseen_aggregate": str(paths["unseen"]),
        "thresholds": {
            "minimum_standard_nominal_safe_success": 0.85,
            "minimum_standard_intervention_safe_success": 0.90,
            "minimum_standard_per_seed_safe_success": 0.80,
            "minimum_strict_safe_success": 0.90,
            "minimum_strict_per_seed_safe_success": 0.80,
            "maximum_strict_safe_success_drop_from_incumbent": 0.05,
            "minimum_mean_unseen_ood_safe_success": 0.80,
            "minimum_unseen_ood_per_seed_safe_success": 0.60,
            "require_all_frozen_unseen_ood_hypotheses": True,
            "require_causal_progress_utility": True,
        },
        "claim_boundary": "synthetic test",
    }
    gate_path = tmp_path / "gate.json"
    write(gate_path, gate)
    return gate_path, paths


def test_v28_final_gate_passes_complete_strong_evidence(tmp_path):
    gate, _ = fixtures(tmp_path)
    output = tmp_path / "result.json"
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "--config", str(gate), "--output", str(output)],
        check=False,
    )
    assert completed.returncode == 0
    assert json.loads(output.read_text())["passed"] is True


def test_v28_final_gate_fails_closed_on_one_weak_unseen_seed(tmp_path):
    gate, paths = fixtures(tmp_path)
    unseen = json.loads(paths["unseen"].read_text())
    unseen["records"][0]["per_seed"][0]["variant_safe_success_rate"] = 0.59
    write(paths["unseen"], unseen)
    output = tmp_path / "result.json"
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "--config", str(gate), "--output", str(output)],
        check=False,
    )
    assert completed.returncode == 1
    result = json.loads(output.read_text())
    assert result["passed"] is False
    assert result["checks"]["unseen_ood_per_seed_safe_success"] is False

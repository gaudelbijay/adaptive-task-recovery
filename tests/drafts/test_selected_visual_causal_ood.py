import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from run_selected_visual_causal_ood import (  # noqa: E402
    evaluation_filename,
    resolve_task,
)


def policy(tmp_path):
    path = tmp_path / "policy.json"
    path.write_text(json.dumps({
        "name": "policy", "seeds": [11, 22, 33],
        "experiments": [{"method": "rgb"}],
    }))
    return path


def selection():
    return {
        "selected": "candidate", "all_candidates_ineligible": False,
        "candidates": [{"label": "candidate", "eligible": True}],
    }


def test_variant_major_seed_minor_task_resolution(tmp_path):
    spec = {
        "policy_configs": {"candidate": str(policy(tmp_path))},
        "variants": [
            {"name": "zero", "progress_head_mode": "zero", "visual_perturbation": "none"},
            {"name": "dim", "progress_head_mode": "normal", "visual_perturbation": "brightness_70"},
        ],
    }
    assert resolve_task(spec, selection(), 0)["seed"] == 11
    task = resolve_task(spec, selection(), 4)
    assert task["seed"] == 22
    assert task["variant"]["name"] == "dim"
    assert task["task_count"] == 6


def test_resolution_fails_closed_without_an_eligible_selection(tmp_path):
    spec = {"policy_configs": {"candidate": str(policy(tmp_path))}, "variants": [{}]}
    invalid = selection()
    invalid["all_candidates_ineligible"] = True
    with pytest.raises(ValueError, match="no eligible policy"):
        resolve_task(spec, invalid, 0)


def test_output_names_are_non_overwriting():
    assert evaluation_filename("nominal", "zero", "none") == (
        "heldout_eval_nominal_progress_zero.json"
    )
    assert evaluation_filename("intervention", "normal", "brightness_70") == (
        "heldout_eval_intervention_visual_brightness_70.json"
    )
    assert evaluation_filename(
        "intervention", "normal", "none", "camera_left_5cm",
    ) == "heldout_eval_intervention_env_camera_left_5cm.json"

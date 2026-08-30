import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def load(name):
    return json.loads((ROOT / "configs" / name).read_text())


def test_v24_strict_extension_retains_every_prior_candidate_once():
    config = load("strict_removal_dual_specialist_shift_action_extension_v12.json")
    labels = [cohort["label"] for cohort in config["cohorts"]]
    assert labels == [
        "strict_stable_visual",
        "dual_specialist_visual",
        "vicreg_dual_specialist_visual",
        "low_variance_vicreg_dual_specialist_visual",
        "bounded_shift_action_dual_specialist_visual",
    ]
    assert len(labels) == len(set(labels))


def test_v24_selector_retains_frozen_thresholds_and_all_candidates():
    prior = load("integrated_visual_selection_v8.json")
    current = load("integrated_visual_selection_v9.json")
    assert current["thresholds"] == prior["thresholds"]
    assert current["candidates"][:-1] == prior["candidates"]
    added = current["candidates"][-1]
    assert added["label"] == "bounded_shift_action_dual_specialist_visual"
    assert added["method"] == "event_reward_dual_specialist_shift_action_visual_ppo"
    assert "before V24 smoke or held-out results" in current["claim_boundary"]

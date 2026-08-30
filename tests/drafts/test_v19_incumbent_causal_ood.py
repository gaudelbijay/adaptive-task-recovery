import json
from pathlib import Path


INCUMBENT = Path("configs/v19_incumbent_causal_ood_v1.json")
FINAL_SELECTOR = Path("configs/selected_visual_causal_ood_v1.json")
SELECTION = Path("results/gates/integrated_visual_selection_v6.json")


def test_incumbent_suite_preserves_frozen_protocol_and_uses_completed_v19_selection():
    incumbent = json.loads(INCUMBENT.read_text())
    final = json.loads(FINAL_SELECTOR.read_text())
    selection = json.loads(SELECTION.read_text())

    for field in (
        "conditions", "episodes", "num_envs", "seed_base",
        "hypothesis_thresholds", "variants",
    ):
        assert incumbent[field] == final[field]
    assert incumbent["selection"] == str(SELECTION)
    assert selection["selected"] == "dual_specialist_visual"
    assert selection["all_candidates_ineligible"] is False
    selected = [
        item for item in selection["candidates"]
        if item["label"] == selection["selected"]
    ]
    assert len(selected) == 1
    assert selected[0]["eligible"] is True
    assert incumbent["policy_configs"]["dual_specialist_visual"] == (
        "configs/visual_recovery_dual_specialist_dagger_v19.json"
    )
    assert incumbent["variants"][0] == {
        "name": "baseline", "progress_head_mode": "normal",
        "visual_perturbation": "none", "environment_profile": "nominal",
    }
    assert len(incumbent["variants"]) == 11
    assert "does not prejudge or replace" in incumbent["claim_boundary"]

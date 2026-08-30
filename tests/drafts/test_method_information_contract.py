import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from build_method_information_contract import build  # noqa: E402


def test_method_information_contract_discloses_inputs_supervision_and_compute():
    manifest = json.loads((
        ROOT / "configs/paper_method_information_contract_v1.json"
    ).read_text(encoding="utf-8"))
    payload = build(manifest)
    assert len(payload["methods"]) == 20
    records = {item["label"]: item for item in payload["methods"]}

    clean = records["Clean learned-progress RGB"]
    assert clean["executed_ppo_interactions_per_seed"] == 39_993_344
    assert clean["dagger_interactions_per_seed"] == 1_920_000
    assert clean["training_only_state_teacher"] is True
    assert clean["training_only_goal_resolution_labels"] is True
    assert "RGB" in clean["deployed_actor_inputs"]
    assert "object pose" not in clean["deployed_actor_inputs"]

    v15 = records["Integrated-teacher RGB V15"]
    v16 = records["VICReg integrated-teacher RGB V16"]
    assert v15["vicreg_variance_coefficient"] == 0.0
    assert v16["vicreg_variance_coefficient"] == 0.01
    assert v16["vicreg_covariance_coefficient"] == 0.001
    assert v15["new_interactions_per_seed"] == v16["new_interactions_per_seed"]
    assert v15["initializer_checkpoint"] == v16["initializer_checkpoint"]
    assert v15["teacher_checkpoint"] == v16["teacher_checkpoint"]
    assert v15["initializer_label"] == "Integrated RGB V13"
    assert v15["teacher_label"] == "Scratch integrated state"
    assert v15["reported_interactions_exclude_upstream_training"] is True

    v19 = records["Dual-specialist RGB V19"]
    assert v19["training_only_state_teacher"] is True
    assert v19["teacher_checkpoint"] is None
    assert v19["nominal_visual_teacher_checkpoint"]
    assert v19["strict_state_teacher_checkpoint"]
    assert v19["dagger_interactions_per_seed"] == 1_920_000
    assert v19["deployed_actor_inputs"] == v15["deployed_actor_inputs"]
    assert v19["reported_interactions_exclude_upstream_training"] is True

    v20 = records["VICReg dual-specialist RGB V20"]
    assert v20["new_interactions_per_seed"] == v19["new_interactions_per_seed"]
    assert v20["initializer_checkpoint"] == v19["initializer_checkpoint"]
    assert v20["nominal_visual_teacher_checkpoint"] == v19["nominal_visual_teacher_checkpoint"]
    assert v20["strict_state_teacher_checkpoint"] == v19["strict_state_teacher_checkpoint"]
    assert v20["vicreg_variance_coefficient"] == 0.01
    assert v20["vicreg_covariance_coefficient"] == 0.001

    state = records["Scratch integrated state"]
    assert state["deployed_actor_inputs"] == "privileged flattened simulator state"
    assert state["training_only_state_teacher"] is False
    assert state["dagger_interactions_per_seed"] == 0


def test_reverse_curriculum_is_not_misreported_as_matched_training_distribution():
    manifest = json.loads((
        ROOT / "configs/paper_method_information_contract_v1.json"
    ).read_text(encoding="utf-8"))
    records = {item["label"]: item for item in build(manifest)["methods"]}
    assert records["Scratch integrated state"]["training_intervention_probability"] == 0.8
    assert records["Reverse-curriculum state"]["training_intervention_probability"] == 0.2
    assert records["Scratch integrated state"]["selection_intervention_probability"] == 0.5
    assert records["Reverse-curriculum state"]["selection_intervention_probability"] == 0.5

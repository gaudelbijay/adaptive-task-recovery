import copy
import json
from pathlib import Path


V19 = Path("configs/visual_recovery_dual_specialist_dagger_v19.json")
V26 = Path("configs/visual_recovery_dual_specialist_no_temporal_v26.json")
PROTOCOL = Path("configs/temporal_ssl_continuation_ablation_v1.json")


def test_v26_changes_only_identity_claim_and_continuation_temporal_weight():
    treatment = json.loads(V19.read_text())
    control = json.loads(V26.read_text())
    assert treatment["seeds"] == control["seeds"] == [9351, 4796, 1788]
    assert treatment["experiments"][0]["temporal_ssl_coefficient"] == 0.01
    assert control["experiments"][0]["temporal_ssl_coefficient"] == 0.0

    left, right = copy.deepcopy(treatment), copy.deepcopy(control)
    for payload in (left, right):
        payload.pop("name")
        payload.pop("claim_boundary")
        payload["experiments"][0].pop("method")
        payload["experiments"][0].pop("temporal_ssl_coefficient")
    assert left == right


def test_ablation_rule_is_frozen_and_preserves_lineage_boundary():
    protocol = json.loads(PROTOCOL.read_text())
    assert protocol["required_training_seeds"] == [9351, 4796, 1788]
    assert protocol["required_episodes_per_condition"] == 768
    assert protocol["minimum_worst_endpoint_improvement"] == 0.03
    assert protocol["maximum_treatment_violation_rate"] == 0.05
    assert "positive lower bound" in protocol["confirmation_rule"]
    assert "upstream checkpoints" in protocol["claim_boundary"]

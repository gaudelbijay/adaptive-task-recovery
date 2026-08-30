import json
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from train_visual_recovery_vicreg_ppo import vicreg_regularization  # noqa: E402


def load(name):
    return json.loads((ROOT / "configs" / name).read_text(encoding="utf-8"))


def test_vicreg_penalty_is_finite_differentiable_and_detects_collapse():
    collapsed = torch.zeros((64, 16), requires_grad=True)
    variance, covariance = vicreg_regularization(collapsed)
    assert torch.isfinite(variance)
    assert torch.isfinite(covariance)
    assert variance > 0.98
    assert covariance == 0
    (variance + covariance).backward()
    assert collapsed.grad is not None
    assert torch.isfinite(collapsed.grad).all()

    torch.manual_seed(7)
    diverse = torch.randn((4096, 16), requires_grad=True)
    diverse_variance, diverse_covariance = vicreg_regularization(diverse)
    assert diverse_variance < 0.02
    assert diverse_covariance < 0.2


def test_v16_is_a_matched_anticollapse_ablation_of_v15():
    v15 = load("visual_recovery_integrated_teacher_dagger_v15.json")
    v16 = load("visual_recovery_vicreg_integrated_teacher_v16.json")
    assert v16["seeds"] == v15["seeds"] == [9351, 4796, 1788]
    left = dict(v15["experiments"][0])
    right = dict(v16["experiments"][0])
    assert right.pop("temporal_variance_coefficient") == 0.01
    assert right.pop("temporal_covariance_coefficient") == 0.001
    left.pop("method")
    right.pop("method")
    assert right == left
    for key in v15:
        if key not in {"name", "experiments", "claim_boundary"}:
            assert v16[key] == v15[key]


def test_vicreg_smoke_executes_complete_rollout_batches_and_is_not_claimable():
    smoke = load("visual_recovery_vicreg_smoke.json")
    task = smoke["experiments"][0]
    batch = task["num_envs"] * task["num_steps"]
    assert task["total_timesteps"] % batch == 0
    assert task["bc_pretrain_updates"] == 50
    assert smoke["seeds"] == [9351]
    assert "never eligible" in smoke["claim_boundary"]


def test_vicreg_wrapper_preserves_24_hour_exact_resume_contract():
    source = (ROOT / "scripts/slurm_visual_recovery_vicreg_ppo.sh").read_text(
        encoding="utf-8"
    )
    assert "train_visual_recovery_vicreg_ppo.py" in source
    assert "#SBATCH --signal=USR1@300" in source
    assert "#SBATCH --signal=B:" not in source
    assert "#SBATCH --requeue" in source
    assert "scontrol requeue" in source


def test_v16_selection_and_representation_ablations_are_frozen():
    selection = load("integrated_visual_selection_v4.json")
    assert [item["label"] for item in selection["candidates"]] == [
        "strict_stable_visual", "integrated_teacher_visual",
        "vicreg_integrated_teacher_visual",
    ]
    assert selection["thresholds"] == load(
        "integrated_state_teacher_gate_v1.json"
    )["thresholds"]
    representation = load("visual_representation_vicreg_ablation_v5.json")
    assert representation["required_training_seeds"] == 3
    assert representation["primary_diagnostic"] == {
        "treatment": "vicreg_integrated_teacher_visual_encoder",
        "control": "integrated_teacher_visual_encoder",
    }


def test_failure_only_v17_v18_are_matched_reverse_teacher_ablations():
    v17 = load("visual_recovery_reverse_teacher_dagger_v17.json")
    v18 = load("visual_recovery_vicreg_reverse_teacher_v18.json")
    left = dict(v17["experiments"][0])
    right = dict(v18["experiments"][0])
    assert right.pop("temporal_variance_coefficient") == 0.01
    assert right.pop("temporal_covariance_coefficient") == 0.001
    left.pop("method")
    right.pop("method")
    assert right == left
    assert "learned_recovery_ppo_v13_integrated_from_strict" in left[
        "bc_teacher_checkpoint"
    ]
    selection = load("integrated_visual_selection_v5.json")
    assert selection["thresholds"] == load(
        "integrated_from_strict_state_teacher_gate_v2.json"
    )["thresholds"]
    assert [item["label"] for item in selection["candidates"]] == [
        "strict_stable_visual", "reverse_teacher_visual",
        "vicreg_reverse_teacher_visual",
    ]
    assert "failure-only" in selection["claim_boundary"]

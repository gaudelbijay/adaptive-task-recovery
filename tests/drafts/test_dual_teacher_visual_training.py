import json
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from train_visual_recovery_dual_teacher_ppo import dual_teacher_strict_route  # noqa: E402
from train_visual_recovery_dual_teacher_vicreg_ppo import (  # noqa: E402
    dual_teacher_strict_route as vicreg_dual_teacher_strict_route,
    vicreg_regularization,
)


def load(name):
    return json.loads((ROOT / "configs" / name).read_text(encoding="utf-8"))


def test_route_uses_unavailable_not_completed_goal():
    observation = {
        "extra": {
            "goal_progress": torch.tensor([[0, 0], [1, 0], [0, 0], [1, 0]]),
            "critic_goal_resolved": torch.tensor([[0, 0], [1, 0], [1, 0], [1, 1]]),
        }
    }
    route = dual_teacher_strict_route(observation)
    assert route.shape == (4, 1)
    assert route[:, 0].tolist() == [False, False, True, True]


def test_full_and_smoke_configs_preserve_actor_and_task_contract():
    full = load("visual_recovery_dual_specialist_dagger_v19.json")
    smoke = load("visual_recovery_dual_specialist_smoke.json")
    full_task, smoke_task = full["experiments"][0], smoke["experiments"][0]
    for task in (full_task, smoke_task):
        assert task["env_id"] == "LearnedRecovery-v3"
        assert task["control_mode"] == "pd_joint_delta_pos"
        assert task["actor_learned_goal_progress"] is True
        assert task["actor_tcp_pose"] is True
        assert task["asymmetric_critic"] is True
        assert "bc_nominal_visual_teacher_checkpoint" in task
        assert "bc_strict_state_teacher_checkpoint" in task
        assert "bc_teacher_checkpoint" not in task
        assert task["env_kwargs"]["intervention_probability"] == 0.8
        assert task["eval_env_kwargs"]["intervention_probability"] == 0.5
    for key in (
        "actor_learned_goal_progress", "actor_tcp_pose", "asymmetric_critic",
        "temporal_ssl_coefficient", "privileged_aux_coefficient",
        "goal_progress_aux_coefficient", "control_mode", "image_size",
    ):
        assert smoke_task[key] == full_task[key]
    assert full_task["total_timesteps"] == 100_000_000
    assert full_task["bc_pretrain_updates"] * full_task["num_envs"] == 1_920_000
    assert full["seeds"] == [9351, 4796, 1788]


def test_dual_teacher_vicreg_is_finite_differentiable_and_uses_same_route():
    latent = torch.randn(32, 256, requires_grad=True)
    variance, covariance = vicreg_regularization(latent)
    assert torch.isfinite(variance)
    assert torch.isfinite(covariance)
    (variance + covariance).backward()
    assert latent.grad is not None
    assert torch.isfinite(latent.grad).all()
    observation = {
        "extra": {
            "goal_progress": torch.tensor([[0, 0], [1, 0]]),
            "critic_goal_resolved": torch.tensor([[1, 0], [1, 0]]),
        }
    }
    assert torch.equal(
        dual_teacher_strict_route(observation),
        vicreg_dual_teacher_strict_route(observation),
    )


def test_v20_is_a_matched_vicreg_ablation_of_v19():
    v19 = load("visual_recovery_dual_specialist_dagger_v19.json")
    v20 = load("visual_recovery_vicreg_dual_specialist_v20.json")
    task19 = dict(v19["experiments"][0])
    task20 = dict(v20["experiments"][0])
    assert task20.pop("temporal_variance_coefficient") == 0.01
    assert task20.pop("temporal_covariance_coefficient") == 0.001
    task19.pop("method")
    task20.pop("method")
    assert task20 == task19
    for key in v19:
        if key not in {"name", "experiments", "claim_boundary"}:
            assert v20[key] == v19[key]
    smoke = load("visual_recovery_vicreg_dual_specialist_smoke.json")
    smoke_task = smoke["experiments"][0]
    for key in (
        "actor_learned_goal_progress", "actor_tcp_pose", "asymmetric_critic",
        "temporal_ssl_coefficient", "temporal_variance_coefficient",
        "temporal_covariance_coefficient", "privileged_aux_coefficient",
        "goal_progress_aux_coefficient", "control_mode", "image_size",
    ):
        assert smoke_task[key] == v20["experiments"][0][key]

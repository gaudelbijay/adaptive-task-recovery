"""Static contract tests for the end-to-end learned recovery benchmark."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
ENVIRONMENT = ROOT / "src/atr/envs/learned_recovery.py"
CONFIG = ROOT / "configs/learned_recovery_ppo_v2.json"
SAFE_CONFIG = ROOT / "configs/learned_recovery_ppo_v6.json"
EVENT_ENVIRONMENT = ROOT / "src/atr/envs/learned_recovery_v3.py"
EVENT_ADAPTIVE_CONFIG = ROOT / "configs/learned_recovery_ppo_v8_event_reward.json"
EVENT_NOMINAL_CONFIG = ROOT / "configs/learned_recovery_ppo_v9_event_reward_nominal.json"
INTEGRATED_STATE_CONFIG = ROOT / "configs/learned_recovery_ppo_v12_integrated_mixture.json"
INTEGRATED_VISUAL_CONFIG = ROOT / "configs/visual_recovery_strict_adaptive_v13_stable.json"


def test_pose_assignment_is_reset_only():
    tree = ast.parse(ENVIRONMENT.read_text(encoding="utf-8"))
    violations = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for child in ast.walk(node):
            if (
                isinstance(child, ast.Call)
                and isinstance(child.func, ast.Attribute)
                and child.func.attr == "set_pose"
                and node.name != "_initialize_episode"
            ):
                violations.append((node.name, child.lineno))
    assert violations == []


def test_intervention_is_force_driven():
    source = ENVIRONMENT.read_text(encoding="utf-8")
    assert "def _before_simulation_step" in source
    assert source.count(".apply_force(") == 4  # GPU and scalar CPU paths, two sweepers


def test_only_the_physical_intervention_target_can_be_skipped():
    source = ENVIRONMENT.read_text(encoding="utf-8")
    assert "def _recognized_unavailable" in source
    assert "physical & target & valid_target[:, None] & intervention_started[:, None]" in source
    tree = ast.parse(source)
    functions = {
        node.name: ast.unparse(node)
        for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
    }
    assert "self._recognized_unavailable()" in functions["_update_task_memory"]
    assert "self._recognized_unavailable()" in functions["compute_dense_reward"]


def test_all_primary_methods_learn_the_same_continuous_control_space():
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    experiments = config["experiments"]
    assert {experiment["method"] for experiment in experiments} == {
        "adaptive_ppo", "privileged_oracle_ppo", "no_intervention_training_ppo"
    }
    assert {experiment["control_mode"] for experiment in experiments} == {
        "pd_joint_delta_pos"
    }
    assert all(experiment["total_timesteps"] == 100_000_000 for experiment in experiments)
    assert len(config["seeds"]) == 3


def test_safe_followup_is_matched_and_annealed():
    config = json.loads(SAFE_CONFIG.read_text(encoding="utf-8"))
    assert config["anneal_lr"] is True
    assert config["selection_failure_penalty"] == 2.0
    assert len(config["experiments"]) == 3
    for experiment in config["experiments"]:
        assert experiment["env_kwargs"]["terminate_on_violation"] is True
        assert experiment["env_kwargs"]["safety_proximity_weight"] == 5.0
        assert experiment["env_kwargs"]["constraint_violation_penalty"] == 20.0
        assert experiment["total_timesteps"] == 100_000_000


def test_runtime_step_rejects_any_pose_assignment(monkeypatch):
    pytest.importorskip("mani_skill")
    import gymnasium as gym
    from mani_skill.utils.structs.actor import Actor
    import atr.envs.learned_recovery_v3  # noqa: F401

    env = gym.make(
        "LearnedRecovery-v3", num_envs=1, obs_mode="state", render_mode=None,
        sim_backend="physx_cpu", control_mode="pd_joint_delta_pos",
        intervention_probability=1.0, onset_step_range=(0, 0),
    )
    try:
        env.reset(seed=4)

        def forbidden(*_args, **_kwargs):
            raise AssertionError("actor pose assignment occurred after reset")

        monkeypatch.setattr(Actor, "set_pose", forbidden)
        for _ in range(5):
            _, _, _, _, info = env.step(env.action_space.sample() * 0)
        assert not bool(info["constraint_violated"].item())
    finally:
        env.close()


def test_v3_is_isolated_and_has_no_persistent_completion_reward():
    source = EVENT_ENVIRONMENT.read_text(encoding="utf-8")
    assert '@register_env("LearnedRecovery-v3"' in source
    assert "class LearnedRecoveryEventRewardEnv(LearnedRecoveryEnv)" in source
    assert "completion_bonus * newly_completed" in source
    assert "self._completed.float().sum" not in source
    assert "set_pose" not in source


def test_v3_stalling_has_zero_task_reward_and_completion_is_one_time():
    pytest.importorskip("torch")
    pytest.importorskip("mani_skill")
    import sys
    import torch

    sys.path.insert(0, str(ROOT / "src"))
    from atr.envs.learned_recovery_v3 import event_progress_reward

    common = {
        "progress_scale": 2.0,
        "completion_bonus": 5.0,
        "success_reward": 10.0,
        "safety_proximity_weight": 5.0,
        "constraint_violation_penalty": 20.0,
    }
    reward = event_progress_reward(
        progress_delta=torch.tensor([0.0, 0.0, 0.0]),
        newly_completed=torch.tensor([0.0, 1.0, 0.0]),
        success=torch.tensor([False, False, True]),
        proximity_risk=torch.zeros(3),
        constraint_violated=torch.zeros(3, dtype=torch.bool),
        **common,
    )
    assert reward.tolist() == [0.0, 5.0, 10.0]


def test_v3_nominal_state_upper_bound_matches_adaptive_training_budget():
    adaptive = json.loads(EVENT_ADAPTIVE_CONFIG.read_text(encoding="utf-8"))
    nominal = json.loads(EVENT_NOMINAL_CONFIG.read_text(encoding="utf-8"))
    left = adaptive["experiments"][0]
    right = nominal["experiments"][0]
    ignored = {"method", "env_kwargs", "eval_env_kwargs"}
    assert {key: value for key, value in left.items() if key not in ignored} == {
        key: value for key, value in right.items() if key not in ignored
    }
    assert adaptive["seeds"] == nominal["seeds"]
    assert right["env_kwargs"]["intervention_probability"] == 0.0
    assert right["eval_env_kwargs"]["intervention_probability"] == 0.0
    for key in (
        "terminate_on_violation", "safety_proximity_weight",
        "constraint_violation_penalty", "progress_reward_scale",
        "completion_bonus", "success_reward",
    ):
        assert left["env_kwargs"][key] == right["env_kwargs"][key]


def test_integrated_state_baseline_matches_v13_distribution_contract():
    state = json.loads(INTEGRATED_STATE_CONFIG.read_text(encoding="utf-8"))
    visual = json.loads(INTEGRATED_VISUAL_CONFIG.read_text(encoding="utf-8"))
    state_task = state["experiments"][0]
    visual_task = visual["experiments"][0]
    assert state_task["env_id"] == visual_task["env_id"] == "LearnedRecovery-v3"
    assert state_task["env_kwargs"] == {
        key: value for key, value in visual_task["env_kwargs"].items()
        if key != "required_goals"
    }
    assert state_task["eval_env_kwargs"] == {
        key: value for key, value in visual_task["eval_env_kwargs"].items()
        if key != "required_goals"
    }
    assert state_task["total_timesteps"] == visual_task["total_timesteps"]
    assert state["num_eval_envs"] == 256
    assert state["selection_failure_penalty"] == visual["selection_failure_penalty"]

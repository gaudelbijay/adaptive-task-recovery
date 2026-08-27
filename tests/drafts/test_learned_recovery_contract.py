"""Static contract tests for the end-to-end learned recovery benchmark."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
ENVIRONMENT = ROOT / "src/atr/envs/learned_recovery.py"
CONFIG = ROOT / "configs/learned_recovery_ppo_v2.json"
SAFE_CONFIG = ROOT / "configs/learned_recovery_ppo_v3.json"


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
    assert len(config["experiments"]) == 3
    for experiment in config["experiments"]:
        assert experiment["env_kwargs"]["terminate_on_violation"] is True
        assert experiment["env_kwargs"]["safety_proximity_weight"] == 2.0
        assert experiment["total_timesteps"] == 100_000_000


def test_runtime_step_rejects_any_pose_assignment(monkeypatch):
    pytest.importorskip("mani_skill")
    import gymnasium as gym
    from mani_skill.utils.structs.actor import Actor
    import atr.envs.learned_recovery  # noqa: F401

    env = gym.make(
        "LearnedRecovery-v1", num_envs=1, obs_mode="state", render_mode=None,
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

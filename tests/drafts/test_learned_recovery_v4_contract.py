"""Contract checks for the mechanism-diverse recovery benchmark."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
ENVIRONMENT = ROOT / "src/atr/envs/learned_recovery_v4.py"
OOD_ENVIRONMENT = ROOT / "src/atr/envs/learned_recovery_v4_ood.py"


def test_v4_has_four_semantically_distinct_interventions():
    source = ENVIRONMENT.read_text(encoding="utf-8")
    assert '"reverse_ejection",' in source
    assert '@register_env("LearnedRecovery-v4"' in source
    assert "temporary_return" in source
    assert "permanently_blocked" in source
    assert "reverse_ejection_active" in source


def test_training_only_control_delay_is_explicit_and_label_free():
    source = ENVIRONMENT.read_text(encoding="utf-8")
    step = source.split("def step(self, action):", 1)[1].split(
        "def _load_scene", 1
    )[0]
    assert "control_delay_steps" in step
    assert "torch.zeros_like(action)" in step
    assert "_intervention_mechanism" not in step


def test_v4_has_renderer_only_ood_profiles():
    source = OOD_ENVIRONMENT.read_text(encoding="utf-8")
    assert '@register_env("LearnedRecovery-v4-OOD"' in source
    assert "camera_left_5cm" in source
    assert "camera_high_5cm" in source
    assert "lighting_parameters" in source


def test_v4_runtime_interventions_do_not_assign_poses():
    tree = ast.parse(ENVIRONMENT.read_text(encoding="utf-8"))
    violations = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef) or node.name == "_initialize_episode":
            continue
        for child in ast.walk(node):
            if (
                isinstance(child, ast.Call)
                and isinstance(child.func, ast.Attribute)
                and child.func.attr == "set_pose"
            ):
                violations.append((node.name, child.lineno))
    assert violations == []


def test_temporary_block_is_never_authorized_as_unavailable():
    tree = ast.parse(ENVIRONMENT.read_text(encoding="utf-8"))
    functions = {
        node.name: ast.unparse(node)
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
    }
    recognized = functions["_recognized_unavailable"]
    assert "PERMANENT_BLOCK" in recognized
    assert "TEMPORARY_BLOCK" not in recognized


def test_v4_cpu_smoke_exposes_mechanism_labels():
    pytest.importorskip("mani_skill")
    import gymnasium as gym
    import atr.envs.learned_recovery_v4  # noqa: F401

    env = gym.make(
        "LearnedRecovery-v4", num_envs=1, obs_mode="state", render_mode=None,
        sim_backend="physx_cpu", control_mode="pd_joint_delta_pos",
        intervention_probability=1.0, intervention_types=("permanent_block",),
        onset_step_range=(0, 0), blocker_force=4.0,
    )
    try:
        env.reset(seed=17)
        info = None
        for _ in range(12):
            _, _, _, _, info = env.step(env.action_space.sample() * 0)
        assert int(info["intervention_mechanism"].item()) == 1
        assert bool(info["permanent_goal_block"].item())
        assert not bool(info["temporary_goal_block"].item())
    finally:
        env.close()

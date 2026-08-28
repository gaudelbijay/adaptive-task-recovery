"""Auditable restrictions for the deployed visual recovery policy."""

from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TRAINER = ROOT / "scripts" / "train_visual_recovery_ppo.py"


def _module():
    return ast.parse(TRAINER.read_text(encoding="utf-8"))


def _assignment_literal(name):
    for node in _module().body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == name:
                    return ast.literal_eval(node.value)
    raise AssertionError(f"assignment not found: {name}")


def test_base_actor_extra_contract_contains_only_instruction():
    assert _assignment_literal("ACTOR_EXTRA_KEYS") == ("instruction",)


def test_deployed_action_api_cannot_receive_privileged_critic_state():
    for node in ast.walk(_module()):
        if isinstance(node, ast.ClassDef) and node.name == "VisualAgent":
            method = next(item for item in node.body if isinstance(item, ast.FunctionDef) and item.name == "get_action")
            assert [argument.arg for argument in method.args.args] == [
                "self", "rgb", "proprio", "deterministic",
            ]
            return
    raise AssertionError("VisualAgent.get_action not found")


def test_actor_proprio_contains_only_robot_state_and_instruction():
    function = next(
        node for node in _module().body
        if isinstance(node, ast.FunctionDef) and node.name == "extract_observation"
    )
    source = ast.unparse(function)
    assert "obs['agent']['qpos']" in source
    assert "obs['agent']['qvel']" in source
    assert "ACTOR_EXTRA_KEYS" in source
    actor_section = source.split("if asymmetric:", 1)[0]
    assert "tcp_pose" in actor_section
    for forbidden in ("goal_progress", "critic_red", "critic_blue", "oracle"):
        assert forbidden not in actor_section


def test_policy_actions_are_tanh_bounded_and_never_posthoc_clipped():
    source = TRAINER.read_text(encoding="utf-8")
    assert "action = torch.tanh(pre_tanh_action)" in source
    rollout = source.split("for step in range(t):", 1)[1].split("with torch.no_grad():", 1)[0]
    assert "torch.clamp(action" not in rollout


def test_checkpoint_declares_restricted_observation_contract():
    source = TRAINER.read_text(encoding="utf-8")
    assert '"rgb_qpos_qvel_instruction_v1"' in source
    assert '"rgb_qpos_qvel_tcp_instruction_v2"' in source

"""Static provenance checks for cross-task state-policy initialization."""

from pathlib import Path


SOURCE = (Path(__file__).resolve().parents[2] / "scripts/train_manipulation_ppo.py").read_text()


def test_initialization_loads_weights_but_reinitializes_optimizer():
    section = SOURCE.split('elif task.get("init_checkpoint")', 1)[1].split(
        "num_envs, num_steps", 1
    )[0]
    assert 'agent.load_state_dict(initialization["agent"], strict=True)' in section
    assert "optimizer.load_state_dict" not in section
    assert '"optimizer_reinitialized": True' in section


def test_initialization_is_hash_tracked():
    assert 'hashlib.sha256(initialization_path.read_bytes()).hexdigest()' in SOURCE


def test_optional_anchor_is_frozen_and_penalizes_actor_drift():
    assert "parameter.requires_grad_(False)" in SOURCE
    assert "anchor_agent.actor_mean(b_obs[mb])" in SOURCE

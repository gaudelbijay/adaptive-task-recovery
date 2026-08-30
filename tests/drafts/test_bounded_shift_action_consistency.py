import ast
import hashlib
import json
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from bounded_shift_action_consistency import (  # noqa: E402
    bounded_action_mean,
    bounded_shift_action_consistency,
)
import train_visual_recovery_dual_teacher_shift_action_ppo as wrapped  # noqa: E402


class TinyAgent(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Linear(4, 3)
        self.actor = torch.nn.Linear(5, 2)
        self.goal_progress_predictor = None

    def encode(self, rgb, augment=False):
        value = rgb.reshape(len(rgb), 4).float()
        if augment:
            value = value.roll(1, dims=1)
        return self.encoder(value)


def test_bounded_actions_and_finite_bounded_loss_receive_gradients():
    torch.manual_seed(7)
    agent = TinyAgent()
    rgb = torch.randn(8, 2, 2)
    proprio = torch.randn(8, 2)
    action = bounded_action_mean(agent, rgb, proprio, augment=True)
    loss = bounded_shift_action_consistency(agent, rgb, proprio)
    assert torch.all(action >= -1) and torch.all(action <= 1)
    assert torch.isfinite(loss) and 0 <= loss <= 1.95
    loss.backward()
    assert agent.encoder.weight.grad is not None
    assert agent.actor.weight.grad is not None


def test_clean_target_is_stopped_but_shifted_branch_is_live():
    torch.manual_seed(11)
    agent = TinyAgent()
    rgb = torch.randn(4, 2, 2)
    proprio = torch.randn(4, 2)
    with torch.no_grad():
        target = bounded_action_mean(agent, rgb, proprio, augment=False)
    shifted = bounded_action_mean(agent, rgb, proprio, augment=True)
    assert target.requires_grad is False
    assert shifted.requires_grad is True


def test_wrapper_replaces_loss_and_records_actual_sources():
    assert wrapped.trainer.drac_policy_consistency is bounded_shift_action_consistency
    sources = wrapped.trainer.SOURCE_SHA256
    assert set(sources) == {
        "trainer", "trainer_wrapper", "base_trainer", "environment",
        "environment_v3", "bounded_shift_action_consistency",
    }
    assert sources["trainer"] == sources["trainer_wrapper"]
    assert sources["bounded_shift_action_consistency"] == hashlib.sha256(
        (ROOT / "scripts/bounded_shift_action_consistency.py").read_bytes()
    ).hexdigest()


def test_runtime_config_and_wrapper_preserve_fail_closed_contract():
    config = json.loads((
        ROOT / "configs/visual_recovery_dual_specialist_shift_action_v24_runtime_smoke.json"
    ).read_text())
    task = config["experiments"][0]
    assert task["augmentation_pad"] == 4
    assert task["drac_policy_coefficient"] == 0.1
    assert task["total_timesteps"] == 262144
    assert config["claim_boundary"].startswith("Failure-only")
    wrapper = (
        ROOT / "scripts/slurm_visual_recovery_dual_teacher_shift_action_ppo.sh"
    ).read_text()
    assert "#SBATCH --time=24:00:00" in wrapper
    assert "#SBATCH --signal=USR1@300" in wrapper
    assert "#SBATCH --requeue" in wrapper
    assert "TRAINING_COMPLETE.json" in wrapper


def test_standard_heldout_evaluator_agent_is_checkpoint_compatible():
    def classes(path):
        tree = ast.parse(path.read_text())
        return {
            node.name: ast.dump(node, include_attributes=False)
            for node in tree.body
            if isinstance(node, ast.ClassDef)
            and node.name in {"RandomShiftsAug", "VisualAgent"}
        }

    evaluator_agent_source = ROOT / "scripts/train_visual_recovery_ppo.py"
    training_agent_source = (
        ROOT / "scripts/train_visual_recovery_dual_teacher_drac_ppo.py"
    )
    assert classes(evaluator_agent_source) == classes(training_agent_source)

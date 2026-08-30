import ast
import json
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from drac_policy_consistency import drac_policy_consistency  # noqa: E402
from train_visual_recovery_dual_teacher_drac_ppo import (  # noqa: E402
    SOURCE_SHA256,
    VisualAgent,
)


def load(name):
    return json.loads((ROOT / "configs" / name).read_text())


def test_drac_trainer_ppo_ratio_uses_only_original_observations():
    path = ROOT / "scripts/train_visual_recovery_dual_teacher_drac_ppo.py"
    source = path.read_text()
    tree = ast.parse(source)
    calls = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "action_and_value"
        and any(keyword.arg == "augment" for keyword in node.keywords)
    ]
    assert len(calls) == 1
    augment = next(keyword.value for keyword in calls[0].keywords if keyword.arg == "augment")
    assert isinstance(augment, ast.Constant) and augment.value is False
    assert "+ drac_coefficient * drac_loss" in source
    assert '"drac_policy_consistency"' in source


def test_drac_source_provenance_includes_separate_loss_module():
    assert set(SOURCE_SHA256) == {
        "trainer", "environment", "environment_v3", "drac_policy_consistency",
    }
    assert SOURCE_SHA256["drac_policy_consistency"] == (
        __import__("hashlib").sha256(
            (ROOT / "scripts/drac_policy_consistency.py").read_bytes()
        ).hexdigest()
    )


def test_frozen_drac_configs_require_policy_kl_and_random_shift():
    for name in (
        "visual_recovery_dual_specialist_drac_v22_smoke.json",
        "visual_recovery_dual_specialist_drac_v22.json",
    ):
        task = load(name)["experiments"][0]
        assert task["augmentation_pad"] == 4
        assert task["drac_policy_coefficient"] == 0.1
        assert task["asymmetric_critic"] is True


def test_failure_only_calibrated_runtime_changes_only_coefficient_and_identity():
    original = load("visual_recovery_dual_specialist_drac_v22_runtime_smoke.json")
    calibrated = load(
        "visual_recovery_dual_specialist_drac_v23_calibrated_runtime_smoke.json"
    )
    assert calibrated["experiments"][0]["drac_policy_coefficient"] == 0.00009
    derivation = calibrated["coefficient_derivation"]
    bound = (
        0.25
        * (abs(derivation["first_policy_loss"]) + 0.5 * derivation["first_value_loss"])
        / derivation["first_unweighted_kl"]
    )
    assert derivation["unrounded_upper_bound"] == pytest.approx(bound)
    assert derivation["selected_coefficient"] <= bound
    assert calibrated["claim_boundary"].startswith("Failure-only")
    for config in (original, calibrated):
        config.pop("name")
        config.pop("claim_boundary")
        config.pop("coefficient_derivation", None)
        task = config["experiments"][0]
        task.pop("method")
        task.pop("drac_policy_coefficient")
    assert original == calibrated


def test_runtime_smoke_exercises_updates_checkpoint_and_final_evaluation():
    config = load("visual_recovery_dual_specialist_drac_v22_runtime_smoke.json")
    task = config["experiments"][0]
    batch = task["num_envs"] * task["num_steps"]
    iterations = task["total_timesteps"] // batch
    assert task["bc_pretrain_updates"] > 0
    assert task["total_timesteps"] <= 262144
    assert config["update_epochs"] > 0
    assert iterations % config["checkpoint_freq"] == 0
    assert iterations % config["eval_freq"] == 0


def test_real_visual_agent_drac_loss_backpropagates():
    torch.manual_seed(5)
    agent = VisualAgent(
        image_size=64, proprio_dim=6, critic_dim=4, action_dim=3,
        asymmetric=True, aug_pad=4, learned_goal_progress=True,
    )
    rgb = torch.randint(0, 256, (8, 64, 64, 3), dtype=torch.uint8)
    proprio = torch.randn(8, 6)
    loss = drac_policy_consistency(agent, rgb, proprio)
    assert torch.isfinite(loss) and loss >= 0
    loss.backward()
    assert agent.encoder[0][0].weight.grad is not None
    assert agent.actor[-1].weight.grad is not None


def test_drac_wrapper_retains_exact_requeue_contract():
    source = (
        ROOT / "scripts/slurm_visual_recovery_dual_teacher_drac_ppo.sh"
    ).read_text()
    assert "#SBATCH --time=24:00:00" in source
    assert "#SBATCH --signal=USR1@300" in source
    assert "#SBATCH --requeue" in source
    assert "TRAINING_COMPLETE.json" in source
    assert "scontrol requeue" in source

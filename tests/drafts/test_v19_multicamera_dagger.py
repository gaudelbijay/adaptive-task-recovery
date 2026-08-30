import ast
import copy
import json
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, str(Path("scripts").resolve()))
from evaluate_visual_recovery_ppo import apply_visual_perturbation  # noqa: E402
from train_v19_multicamera_dagger import apply_sensor_shift  # noqa: E402
from atr.envs.learned_recovery_v3_multicamera import CAMERA_EYES  # noqa: E402
from atr.envs.learned_recovery_v3_ood import camera_eye  # noqa: E402


ENVIRONMENT = Path("src/atr/envs/learned_recovery_v3_multicamera.py")
TRAINER = Path("scripts/train_v19_multicamera_dagger.py")
SMOKE = Path("configs/visual_recovery_v19_multicamera_dagger_v31_smoke.json")
FULL = Path("configs/visual_recovery_v19_multicamera_dagger_v31.json")
DEVELOPMENT = Path("configs/v31_smoke_development_ood_v1.json")
GATE = Path("configs/v31_multicamera_dagger_smoke_gate_v1.json")


def test_multicamera_environment_changes_only_sensor_configuration():
    tree = ast.parse(ENVIRONMENT.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef))
    assert ast.unparse(cls.bases[0]) == "LearnedRecoveryEventRewardEnv"
    methods = [node.name for node in cls.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
    assert methods == ["_default_sensor_configs"]
    assert np.allclose(CAMERA_EYES["camera_left_5cm"], camera_eye("camera_left_5cm"))
    assert np.allclose(CAMERA_EYES["camera_high_5cm"], camera_eye("camera_high_5cm"))


def test_v31_uses_one_full_episode_simulator_and_v19_only_action_teacher():
    source = TRAINER.read_text()
    assert "StateAgent" not in source
    assert "paired_state_error" not in source
    assert 'teacher.get_action(base_rgb, proprio, deterministic=True)' in source
    assert 'views["camera_left_5cm"]' not in source  # generic camera-key loop
    tree = ast.parse(source)
    env_step_calls = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute) and node.func.attr == "step"
        and node.args and ast.unparse(node.args[0]) == "executed"
    ]
    assert len(env_step_calls) == 1


def test_v31_sensor_shifts_match_observed_evaluator():
    rgb = torch.arange(2 * 16 * 16 * 3, dtype=torch.int64)
    rgb = (rgb % 256).to(torch.uint8).reshape(2, 16, 16, 3)
    for mode in ("pixel_shift_right_4", "brightness_70", "warm_color_shift"):
        assert torch.equal(apply_sensor_shift(rgb, mode), apply_visual_perturbation(rgb, mode))


def test_v31_smoke_full_and_gate_contracts_are_matched():
    smoke, full = json.loads(SMOKE.read_text()), json.loads(FULL.read_text())
    left, right = copy.deepcopy(smoke), copy.deepcopy(full)
    for payload in (left, right):
        payload.pop("name")
        payload.pop("seeds")
        payload.pop("claim_boundary")
        payload["experiments"][0].pop("method")
        payload["experiments"][0].pop("dagger_updates")
        payload["experiments"][0].pop("total_timesteps")
    assert left == right
    for config in (smoke, full):
        task = config["experiments"][0]
        assert task["total_timesteps"] == task["dagger_updates"] * task["num_envs"]
    development, gate = json.loads(DEVELOPMENT.read_text()), json.loads(GATE.read_text())
    assert len(development["variants"]) == 11
    assert development["seed_base"] == 81000000
    assert gate["thresholds"]["minimum_worst_ood_safe_success"] == 0.25
    assert gate["thresholds"]["minimum_mean_ood_safe_success_improvement"] == 0.20

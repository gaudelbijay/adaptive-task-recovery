import json
from pathlib import Path

import numpy as np
import torch

import evaluate_v35_visual_recovery_unseen_ood as unseen


def test_d176_config_uses_full_three_seed_policy_and_only_new_domains():
    spec = json.loads(Path("configs/v35_confirmatory_unseen_visual_ood_v1.json").read_text())
    policy = json.loads(Path(spec["policy_configs"]["translation_repair"]).read_text())
    assert policy["seeds"] == [9351, 4796, 1788]
    assert spec["selection"] == "configs/v35_full_routing_selection_v1.json"
    names = {variant["name"] for variant in spec["variants"]}
    assert names == {
        "baseline", "progress_cyclic_shift", "subpixel_shift_left_1_5",
        "rotation_clockwise_2deg", "scale_95", "camera_back_3cm",
        "camera_roll_right_2deg", "lighting_cool", "lighting_back_key",
    }


def test_camera_back_profile_moves_exactly_three_centimeters():
    nominal = np.asarray([0.45, 0.0, 0.72])
    shifted = unseen.camera_eye("camera_back_3cm")
    assert np.isclose(np.linalg.norm(shifted - nominal), 0.03)


def test_new_sensor_perturbations_are_deterministic_and_shape_preserving():
    image = torch.arange(64 * 64 * 3, dtype=torch.int64).remainder(256).to(torch.uint8)
    image = image.reshape(1, 64, 64, 3)
    for mode in unseen.NEW_PERTURBATIONS:
        first = unseen.apply_visual_perturbation(image, mode)
        second = unseen.apply_visual_perturbation(image, mode)
        assert first.shape == image.shape
        assert first.dtype == image.dtype
        assert torch.equal(first, second)
        assert not torch.equal(first, image)


def test_roll_quaternion_multiplication_preserves_unit_norm():
    identity = np.asarray([[1.0, 0.0, 0.0, 0.0]])
    angle = np.deg2rad(2.0)
    roll = np.asarray([np.cos(angle / 2), np.sin(angle / 2), 0.0, 0.0])
    result = unseen._quaternion_multiply(identity, roll)
    assert np.isclose(np.linalg.norm(result), 1.0)
    assert np.allclose(result, roll)


def test_batched_camera_pose_components_flatten_to_sapien_shapes():
    position = unseen._flat_float32(torch.tensor([[0.45, 0.0, 0.72]]), 3, "position")
    orientation = unseen._flat_float32(
        torch.tensor([[1.0, 0.0, 0.0, 0.0]]), 4, "orientation",
    )
    assert position.shape == (3,)
    assert orientation.shape == (4,)
    assert position.dtype == np.float32
    assert orientation.dtype == np.float32

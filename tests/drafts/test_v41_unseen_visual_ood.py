import json
from pathlib import Path
import sys

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import evaluate_v41_visual_recovery_unseen_ood as module


def test_frozen_suite_and_evaluator_extensions_match_exactly():
    config = json.loads((ROOT / "configs/v36_confirmatory_unseen_visual_ood_v1.json").read_text())
    perturbations = {v["visual_perturbation"] for v in config["variants"]} - {"none"}
    profiles = {v["environment_profile"] for v in config["variants"]} - {"nominal"}
    assert perturbations == set(module.NEW_PERTURBATIONS)
    assert profiles == set(module.NEW_PROFILES)
    assert config["seed_base"] == 117000000


def test_sensor_transforms_are_deterministic_shape_preserving_and_distinct():
    image = torch.arange(2 * 64 * 64 * 3, dtype=torch.int64).remainder(256).to(torch.uint8)
    image = image.reshape(2, 64, 64, 3)
    outputs = [module.apply_visual_perturbation(image, mode) for mode in module.NEW_PERTURBATIONS]
    for output in outputs:
        assert output.shape == image.shape
        assert output.dtype == image.dtype
    assert all(torch.equal(a, b) for a, b in zip(
        outputs, [module.apply_visual_perturbation(image, mode) for mode in module.NEW_PERTURBATIONS],
    ))
    assert len({output.numpy().tobytes() for output in outputs}) == len(outputs)


def test_camera_profile_combines_right_and_four_centimeter_back_displacements():
    nominal = np.asarray([0.45, 0.0, 0.72])
    target = np.asarray([0.05, 0.0, 0.04])
    shifted = module.camera_eye("camera_right_back_4cm")
    assert shifted[1] == -0.04
    radial_component = shifted.copy()
    radial_component[1] = 0.0
    assert np.isclose(np.linalg.norm(radial_component - target) - np.linalg.norm(nominal - target), 0.04)


def test_lighting_profiles_are_distinct_and_non_nominal():
    bright = module.lighting_parameters("lighting_bright_side")
    green = module.lighting_parameters("lighting_green_ambient")
    assert bright != green
    assert green[0][1] > green[0][0]

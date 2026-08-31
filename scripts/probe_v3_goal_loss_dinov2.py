#!/usr/bin/env python3
"""Fail-fast probe for reference-conditioned visual goal-loss perception.

This is deliberately not a controller result.  It asks whether a frozen,
self-supervised DINOv2 representation can separate which requested object was
physically removed, including under renderer-native camera and lighting shifts.
The probe is trained only on canonical renders.  A positive result is required
before building the proposed irreversible-belief policy around this signal.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import mani_skill.envs  # noqa: F401
import atr.envs.learned_recovery_v3  # noqa: F401
import atr.envs.learned_recovery_v3_ood  # noqa: F401


PROFILES = ("nominal", "camera_left_5cm", "camera_high_5cm", "lighting_dim", "lighting_warm")


def preprocess(rgb: torch.Tensor) -> torch.Tensor:
    image = rgb.permute(0, 3, 1, 2).float().div(255.0)
    image = F.interpolate(image, size=(224, 224), mode="bilinear", align_corners=False)
    mean = image.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = image.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    return (image - mean) / std


@torch.inference_mode()
def dino_tokens(model, rgb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    features = model.forward_features(preprocess(rgb))
    return features["x_norm_clstoken"], features["x_norm_patchtokens"]


def goal_condition(features: torch.Tensor) -> torch.Tensor:
    """Create explicit feature-by-goal interactions for a linear probe.

    Concatenating an image vector and one-hot goal only gives an additive
    classifier.  It cannot represent that the same red/blue visual difference
    has opposite labels depending on which goal is queried.  Separate blocks
    preserve a deliberately linear head while making that interaction explicit.
    """

    batch, width = features.shape
    eye = torch.eye(2, device=features.device, dtype=features.dtype)
    blocks = (
        features[:, None, None, :]
        * eye[None, :, :, None]
    ).reshape(batch, 2, 2 * width)
    return torch.cat((blocks, eye[None, :, :].expand(batch, -1, -1)), dim=2)


def make_env(num_envs: int, profile: str):
    kwargs = dict(
        num_envs=num_envs,
        obs_mode="rgb",
        render_mode=None,
        sim_backend="physx_cuda",
        control_mode="pd_joint_delta_pos",
        reward_mode="normalized_dense",
        reconfiguration_freq=1,
        asymmetric_critic_observation=True,
        intervention_probability=1.0,
        onset_step_range=(0, 0),
        intervention_force=6.0,
        intervention_steps=24,
        terminate_on_violation=True,
        safety_proximity_weight=5.0,
        constraint_violation_penalty=20.0,
        progress_reward_scale=2.0,
        completion_bonus=5.0,
        success_reward=10.0,
        vision_camera_size=64,
    )
    env_id = "LearnedRecovery-v3"
    if profile != "nominal":
        env_id = "LearnedRecovery-v3-OOD"
        kwargs["visual_domain_profile"] = profile
    env = gym.make(env_id, **kwargs)
    if isinstance(env.action_space, gym.spaces.Dict):
        env = FlattenActionSpaceWrapper(env)
    return ManiSkillVectorEnv(env, num_envs, ignore_terminations=True, record_metrics=False)


@torch.inference_mode()
def collect(model, *, profile: str, seed_base: int, batches: int, num_envs: int):
    """Collect paired before/after frames; each frame contributes both goals."""

    env = make_env(num_envs, profile)
    dino_rows: list[np.ndarray] = []
    pixel_rows: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    try:
        for batch in range(batches):
            obs, _ = env.reset(seed=seed_base + batch * num_envs)
            reference = obs["sensor_data"]["base_camera"]["rgb"].clone()
            action = torch.zeros(
                (num_envs,) + env.single_action_space.shape,
                device=reference.device, dtype=torch.float32,
            )
            for _ in range(30):
                obs, _, _, _, _ = env.step(action)
            current = obs["sensor_data"]["base_camera"]["rgb"]
            unavailable = (
                obs["extra"]["critic_goal_resolved"].bool()
                & ~obs["extra"]["goal_progress"].bool()
            )
            if not bool((unavailable.sum(dim=1) == 1).all()):
                raise RuntimeError("strict collection did not remove exactly one goal")

            ref_cls, ref_patch = dino_tokens(model, reference)
            cur_cls, cur_patch = dino_tokens(model, current)
            cls_delta = cur_cls - ref_cls
            patch_delta = cur_patch - ref_patch
            common = torch.cat(
                (cls_delta, cls_delta.abs(), patch_delta.mean(dim=1),
                 patch_delta.abs().mean(dim=1), patch_delta.abs().amax(dim=1)),
                dim=1,
            )
            dino = goal_condition(common).reshape(num_envs * 2, -1)

            # A low-capacity pixel baseline receives the same before/after
            # information and goal identity at 8x8 resolution.
            ref_small = F.interpolate(
                reference.permute(0, 3, 1, 2).float().div(255), size=(8, 8),
                mode="bilinear", align_corners=False,
            ).flatten(1)
            cur_small = F.interpolate(
                current.permute(0, 3, 1, 2).float().div(255), size=(8, 8),
                mode="bilinear", align_corners=False,
            ).flatten(1)
            pixel_delta = cur_small - ref_small
            pixel_common = torch.cat((pixel_delta, pixel_delta.abs()), dim=1)
            pixel = goal_condition(pixel_common).reshape(num_envs * 2, -1)
            dino_rows.append(dino.cpu().numpy())
            pixel_rows.append(pixel.cpu().numpy())
            labels.append(unavailable.reshape(-1).cpu().numpy().astype(np.int64))
    finally:
        env.close()
    return np.concatenate(dino_rows), np.concatenate(pixel_rows), np.concatenate(labels)


def fit(features: np.ndarray, labels: np.ndarray):
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(C=0.1, max_iter=2000, class_weight="balanced"),
    ).fit(features, labels)


def score(model, features: np.ndarray, labels: np.ndarray) -> dict:
    probability = model.predict_proba(features)[:, 1]
    prediction = probability >= 0.5
    return {
        "examples": int(len(labels)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, prediction)),
        "roc_auc": float(roc_auc_score(labels, probability)),
        "brier": float(brier_score_loss(labels, probability)),
        "positive_recall": float(prediction[labels == 1].mean()),
        "negative_recall": float((~prediction[labels == 0]).mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/probes/v3_goal_loss_dinov2_v1.json")
    parser.add_argument("--train-batches", type=int, default=8)
    parser.add_argument("--eval-batches", type=int, default=4)
    parser.add_argument("--num-envs", type=int, default=32)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("the V3 DINOv2 probe requires a CUDA renderer")
    model = torch.hub.load(
        "facebookresearch/dinov2", "dinov2_vits14", verbose=False,
    ).eval().cuda()
    train_dino, train_pixel, train_labels = collect(
        model, profile="nominal", seed_base=141_000_000,
        batches=args.train_batches, num_envs=args.num_envs,
    )
    dino_probe = fit(train_dino, train_labels)
    pixel_probe = fit(train_pixel, train_labels)
    evaluations = {}
    for index, profile in enumerate(PROFILES):
        dino, pixel, labels = collect(
            model, profile=profile, seed_base=142_000_000 + index * 1_000_000,
            batches=args.eval_batches, num_envs=args.num_envs,
        )
        evaluations[profile] = {
            "dinov2": score(dino_probe, dino, labels),
            "pixel": score(pixel_probe, pixel, labels),
        }
    payload = {
        "schema_version": 1,
        "protocol": "reference-conditioned frozen-DINOv2 goal-loss probe",
        "train_profile": "nominal",
        "train_examples": int(len(train_labels)),
        "profiles": evaluations,
        "claim_boundary": (
            "Perception-only fail-fast diagnostic on step-0 physical removal; "
            "not a controller, recovery, general-language, or real-robot result."
        ),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Held-out-mechanism probe for temporal visual goal feasibility.

Training sees forward object ejection, permanent and temporary goal blockade,
plus nominal episodes.  Reverse ejection uses an independently actuated
sweeper and is entirely held out from training.  A
query is one goal in one episode.  The positive label means that goal is
causally unavailable; temporary visual obstruction is always a negative.

This is a fail-fast representation test, not a controller result.  It reports
each observation horizon separately so accuracy gained by waiting for a
temporary intervention to clear cannot be mistaken for instantaneous failure
recognition.
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
import atr.envs.learned_recovery_v4  # noqa: F401
import atr.envs.learned_recovery_v4_ood  # noqa: F401


PROFILES = (
    "nominal", "camera_left_5cm", "camera_high_5cm",
    "lighting_dim", "lighting_warm",
)
HORIZONS = (1, 4, 8, 16, 32, 48)


def preprocess(rgb: torch.Tensor) -> torch.Tensor:
    image = rgb.permute(0, 3, 1, 2).float().div(255.0)
    image = F.interpolate(image, size=(224, 224), mode="bilinear", align_corners=False)
    mean = image.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = image.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    return (image - mean) / std


@torch.inference_mode()
def visual_deltas(model, reference: torch.Tensor, current: torch.Tensor) -> dict[str, torch.Tensor]:
    ref = model.forward_features(preprocess(reference))
    cur = model.forward_features(preprocess(current))
    cls = cur["x_norm_clstoken"] - ref["x_norm_clstoken"]
    patch = cur["x_norm_patchtokens"] - ref["x_norm_patchtokens"]
    global_delta = torch.cat(
        (cls, cls.abs(), patch.mean(dim=1), patch.abs().mean(dim=1)), dim=1
    )
    side = int(round(patch.shape[1] ** 0.5))
    if side * side != patch.shape[1]:
        raise ValueError("DINO patch tokens do not form a square grid")
    energy = patch.abs().mean(dim=2).reshape(patch.shape[0], side, side)
    # Collapse the source-to-destination x axis while preserving the task's
    # red/blue y lanes.  This encodes the desired invariance: a change at a
    # queried object's source and a change at its goal are comparable evidence.
    axis_delta = torch.cat((energy.mean(dim=2), energy.amax(dim=2)), dim=1)
    return {"dinov2": global_delta, "dinov2_axis": axis_delta}


def pixel_deltas(reference: torch.Tensor, current: torch.Tensor) -> dict[str, torch.Tensor]:
    def small(image):
        return F.interpolate(
            image.permute(0, 3, 1, 2).float().div(255), size=(8, 8),
            mode="bilinear", align_corners=False,
        ).flatten(1)

    delta = small(current) - small(reference)
    global_delta = torch.cat((delta, delta.abs()), dim=1)
    image_delta = (current.float() - reference.float()).abs().mean(dim=3)
    axis_delta = torch.cat(
        (image_delta.mean(dim=2), image_delta.amax(dim=2)), dim=1
    ).div(255.0)
    return {"pixel": global_delta, "pixel_axis": axis_delta}


def goal_condition(features: torch.Tensor) -> torch.Tensor:
    """Give a linear head an explicit feature-by-goal interaction."""
    batch, width = features.shape
    eye = torch.eye(2, device=features.device, dtype=features.dtype)
    blocks = (features[:, None, None, :] * eye[None, :, :, None]).reshape(
        batch, 2, 2 * width
    )
    return torch.cat((blocks, eye[None].expand(batch, -1, -1)), dim=2)


def make_env(num_envs: int, kind: str, profile: str):
    probability = 0.0 if kind == "nominal" else 1.0
    kwargs = dict(
        num_envs=num_envs, obs_mode="rgb", render_mode=None,
        sim_backend="physx_cuda", control_mode="pd_joint_delta_pos",
        reward_mode="normalized_dense", reconfiguration_freq=1,
        asymmetric_critic_observation=True,
        intervention_probability=probability,
        intervention_types=("ejection",) if kind == "nominal" else (kind,),
        onset_step_range=(0, 0), intervention_force=6.0,
        intervention_steps=24, blocker_force=4.0,
        blocker_return_force=5.0, blocker_return_delay_steps=30,
        terminate_on_violation=True, safety_proximity_weight=5.0,
        constraint_violation_penalty=20.0, vision_camera_size=64,
    )
    env_id = "LearnedRecovery-v4"
    if profile != "nominal":
        env_id = "LearnedRecovery-v4-OOD"
        kwargs["visual_domain_profile"] = profile
    env = gym.make(env_id, **kwargs)
    if isinstance(env.action_space, gym.spaces.Dict):
        env = FlattenActionSpaceWrapper(env)
    return ManiSkillVectorEnv(
        env, num_envs, ignore_terminations=True, record_metrics=False
    )


@torch.inference_mode()
def collect(
    model, *, kind: str, profile: str, seed_base: int,
    batches: int, num_envs: int,
) -> tuple[dict[int, dict[str, np.ndarray]], np.ndarray]:
    families = ("dinov2", "dinov2_axis", "pixel", "pixel_axis")
    rows = {h: {family: [] for family in families} for h in HORIZONS}
    label_rows: list[np.ndarray] = []
    env = make_env(num_envs, kind, profile)
    try:
        for batch in range(batches):
            obs, _ = env.reset(seed=seed_base + batch * num_envs)
            reference = obs["sensor_data"]["base_camera"]["rgb"].clone()
            action = torch.zeros(
                (num_envs,) + env.single_action_space.shape,
                device=reference.device, dtype=torch.float32,
            )
            history: dict[str, list[torch.Tensor]] = {family: [] for family in families}
            for step in range(1, max(HORIZONS) + 1):
                obs, _, _, _, _ = env.step(action)
                if step not in HORIZONS:
                    continue
                current = obs["sensor_data"]["base_camera"]["rgb"]
                deltas = {
                    **visual_deltas(model, reference, current),
                    **pixel_deltas(reference, current),
                }
                for family, delta in deltas.items():
                    history[family].append(delta)
                    rows[step][family].append(
                        goal_condition(torch.cat(history[family], dim=1))
                        .reshape(num_envs * 2, -1).cpu().numpy()
                    )

            unavailable = (
                obs["extra"]["critic_goal_resolved"].bool()
                & ~obs["extra"]["goal_progress"].bool()
            )
            expected = kind in ("ejection", "reverse_ejection", "permanent_block")
            if expected and not bool((unavailable.sum(dim=1) == 1).all()):
                raise RuntimeError(f"{kind} failed to make exactly one goal unavailable")
            if not expected and bool(unavailable.any()):
                raise RuntimeError(f"{kind} incorrectly authorized goal skipping")
            label_rows.append(unavailable.reshape(-1).cpu().numpy().astype(np.int64))
    finally:
        env.close()
    packed = {
        horizon: {
            family: np.concatenate(chunks)
            for family, chunks in families.items()
        }
        for horizon, families in rows.items()
    }
    return packed, np.concatenate(label_rows)


def fit(features: np.ndarray, labels: np.ndarray):
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(C=0.03, max_iter=2000, class_weight="balanced"),
    ).fit(features, labels)


def score(model, features: np.ndarray, labels: np.ndarray) -> dict:
    probability = model.predict_proba(features)[:, 1]
    prediction = probability >= 0.5
    positives = labels == 1
    negatives = ~positives
    return {
        "examples": int(len(labels)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, prediction)),
        "roc_auc": (
            float(roc_auc_score(labels, probability))
            if len(np.unique(labels)) == 2 else None
        ),
        "brier": float(brier_score_loss(labels, probability)),
        "positive_recall": float(prediction[positives].mean()) if positives.any() else None,
        "negative_recall": float((~prediction[negatives]).mean()) if negatives.any() else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/probes/v4_temporal_feasibility_v1.json")
    parser.add_argument("--train-batches", type=int, default=6)
    parser.add_argument("--eval-batches", type=int, default=4)
    parser.add_argument("--num-envs", type=int, default=32)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("the V4 temporal probe requires CUDA")
    model = torch.hub.load(
        "facebookresearch/dinov2", "dinov2_vits14", verbose=False,
    ).eval().cuda()

    training_parts = []
    training_labels = []
    training_kinds = ("nominal", "ejection", "permanent_block", "temporary_block")
    for index, kind in enumerate(training_kinds):
        features, labels = collect(
            model, kind=kind, profile="nominal",
            seed_base=151_000_000 + index * 1_000_000,
            batches=args.train_batches, num_envs=args.num_envs,
        )
        training_parts.append(features)
        training_labels.append(labels)
    labels = np.concatenate(training_labels)
    probes = {
        horizon: {
            family: fit(
                np.concatenate([part[horizon][family] for part in training_parts]), labels
            )
            for family in ("dinov2", "dinov2_axis", "pixel", "pixel_axis")
        }
        for horizon in HORIZONS
    }

    evaluations = {}
    eval_kinds = (
        "ejection", "permanent_block", "temporary_block", "reverse_ejection",
    )
    for profile_index, profile in enumerate(PROFILES):
        evaluations[profile] = {}
        for kind_index, kind in enumerate(eval_kinds):
            features, test_labels = collect(
                model, kind=kind, profile=profile,
                seed_base=(160_000_000 + profile_index * 10_000_000
                           + kind_index * 1_000_000),
                batches=args.eval_batches, num_envs=args.num_envs,
            )
            evaluations[profile][kind] = {
                str(horizon): {
                    family: score(probes[horizon][family], features[horizon][family], test_labels)
                    for family in ("dinov2", "dinov2_axis", "pixel", "pixel_axis")
                }
                for horizon in HORIZONS
            }

    payload = {
        "schema_version": 1,
        "protocol": "reference-conditioned temporal feasibility; reverse ejection held out",
        "training_interventions": list(training_kinds),
        "heldout_intervention": "reverse_ejection",
        "horizons": list(HORIZONS),
        "train_goal_queries": int(len(labels)),
        "profiles": evaluations,
        "claim_boundary": (
            "Perception-only mechanism-transfer diagnostic. It does not establish "
            "closed-loop recovery, completed-goal recognition, or real-robot transfer."
        ),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

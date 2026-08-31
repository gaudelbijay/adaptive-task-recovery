#!/usr/bin/env python3
"""Learn visual physical state, then close the loop with the frozen V4 policy."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mani_skill.utils.common import flatten_state_dict

from probe_v4_temporal_feasibility import preprocess
from train_manipulation_ppo import Agent as StateAgent
from train_v4_dino_permanent_student import actor_inputs, make_env
from train_v4_permanent_visual_dagger import (
    reconstruct_v4_state_teacher_observation,
)


FIELDS = (
    ("red_cube_pose", "critic_red_cube_pose", 7),
    ("blue_cube_pose", "critic_blue_cube_pose", 7),
    ("red_goal_pos", "critic_red_goal_pos", 3),
    ("blue_goal_pos", "critic_blue_goal_pos", 3),
    ("red_sweeper_pose", "critic_red_sweeper_pose", 7),
    ("blue_sweeper_pose", "critic_blue_sweeper_pose", 7),
    ("protected_pose", "critic_protected_pose", 7),
    ("red_goal_blocker_pose", "critic_red_goal_blocker_pose", 7),
    ("blue_goal_blocker_pose", "critic_blue_goal_blocker_pose", 7),
)


def target_state(obs):
    return torch.cat([obs["extra"][critic] for _, critic, _ in FIELDS], dim=1)


def target_scale(device):
    values = []
    for _, _, width in FIELDS:
        values.extend(([5.0, 5.0, 5.0] + [1.0] * (width - 3)))
    return torch.tensor(values, device=device)


def compose_teacher_observation(obs, predicted):
    extra = obs["extra"]
    fields = {}
    start = 0
    for name, _, width in FIELDS:
        fields[name] = predicted[:, start:start + width]
        start += width
    state = {
        "agent": obs["agent"],
        "extra": {
            "tcp_pose": extra["tcp_pose"],
            "instruction": extra["instruction"],
            "goal_progress": extra["critic_goal_resolved"].float(),
            **fields,
        },
    }
    return flatten_state_dict(state, use_torch=True)


class SpatialDinoStateEstimator(nn.Module):
    def __init__(self, proprio_dim: int):
        super().__init__()
        self.spatial = nn.Sequential(
            nn.Conv2d(384, 32, 1), nn.ReLU(), nn.AdaptiveAvgPool2d((8, 8)),
        )
        self.cls = nn.Sequential(nn.Linear(384, 128), nn.ReLU())
        width = 32 * 8 * 8 + 128 + proprio_dim + 2
        self.head = nn.Sequential(
            nn.Linear(width, 512), nn.ReLU(),
            nn.Linear(512, 512), nn.ReLU(),
            nn.Linear(512, sum(width for _, _, width in FIELDS)),
        )

    def forward(self, features, proprio, progress):
        patch = features["x_norm_patchtokens"]
        side = int(round(patch.shape[1] ** 0.5))
        patch = patch.transpose(1, 2).reshape(patch.shape[0], 384, side, side)
        spatial = self.spatial(patch).flatten(1)
        cls = self.cls(features["x_norm_clstoken"])
        return self.head(torch.cat((spatial, cls, proprio, progress), dim=1))


@torch.inference_mode()
def evaluate(estimator, teacher, backbone, env, steps, scale):
    obs, _ = env.reset(seed=247_000_000)
    success = torch.zeros(env.num_envs, dtype=torch.bool, device="cuda")
    violation = torch.zeros_like(success)
    errors = []
    for _ in range(steps):
        rgb, proprio, progress = actor_inputs(obs)
        features = backbone.forward_features(preprocess(rgb))
        predicted = estimator(features, proprio, progress) / scale
        errors.append((predicted - target_state(obs)).abs().mean(dim=0))
        action = teacher.get_action(
            compose_teacher_observation(obs, predicted), deterministic=True,
        ).clamp(-1, 1)
        obs, _, _, _, info = env.step(action)
        success |= info["success"].bool()
        violation |= info["constraint_violated"].bool()
    error = torch.stack(errors).mean()
    return {
        "successes": int(success.sum()), "episodes": int(env.num_envs),
        "success_rate": float(success.float().mean()),
        "violations": int(violation.sum()),
        "violation_rate": float(violation.float().mean()),
        "mean_absolute_state_error": float(error),
    }


def atomic_save(payload, path):
    path = Path(path); temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    torch.save(payload, temporary); os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/v4_dino_state_estimator_pilot.json")
    parser.add_argument("--output", default="results/v4_dino_state_estimator_pilot")
    args = parser.parse_args(); config = json.loads(Path(args.config).read_text())
    torch.manual_seed(config["seed"]); np.random.seed(config["seed"])
    train_env = make_env(config["num_envs"]); eval_env = make_env(config["eval_envs"])
    teacher_path = Path(config["teacher_checkpoint"])
    checkpoint = torch.load(teacher_path, map_location="cuda", weights_only=False)
    obs, _ = train_env.reset(seed=config["seed"])
    action_dim = int(np.prod(train_env.single_action_space.shape))
    teacher_obs = reconstruct_v4_state_teacher_observation(obs)
    teacher = StateAgent(teacher_obs.shape[1], action_dim).cuda()
    teacher.load_state_dict(checkpoint["agent"]); teacher.eval()
    rgb, proprio, progress = actor_inputs(obs)
    backbone = torch.hub.load(
        "facebookresearch/dinov2", "dinov2_vits14", verbose=False,
    ).eval().cuda()
    for parameter in backbone.parameters(): parameter.requires_grad_(False)
    estimator = SpatialDinoStateEstimator(proprio.shape[1]).cuda()
    optimizer = torch.optim.AdamW(estimator.parameters(), lr=config["learning_rate"])
    scale = target_scale("cuda")
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    history = output / "metrics.jsonl"; best_score = -float("inf")
    try:
        for update in range(1, config["updates"] + 1):
            rgb, proprio, progress = actor_inputs(obs)
            with torch.no_grad():
                features = backbone.forward_features(preprocess(rgb))
                exact = target_state(obs)
                teacher_action = teacher.get_action(
                    reconstruct_v4_state_teacher_observation(obs), deterministic=True,
                ).clamp(-1, 1)
            predicted_scaled = estimator(features, proprio, progress)
            loss = F.smooth_l1_loss(predicted_scaled, exact * scale)
            optimizer.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(estimator.parameters(), 1.0); optimizer.step()
            predicted = predicted_scaled.detach() / scale
            composed_action = teacher.get_action(
                compose_teacher_observation(obs, predicted), deterministic=True,
            ).clamp(-1, 1)
            probability = config["student_rollout_max"] * update / config["updates"]
            choose = torch.rand(config["num_envs"], device="cuda") < probability
            obs, _, _, _, _ = train_env.step(torch.where(
                choose[:, None], composed_action, teacher_action,
            ))
            if update == 1 or update % config["eval_frequency"] == 0:
                metrics = evaluate(
                    estimator, teacher, backbone, eval_env, config["eval_steps"], scale,
                )
                metrics.update({"update": update, "state_loss": float(loss.detach())})
                with history.open("a") as handle: handle.write(json.dumps(metrics) + "\n")
                score = metrics["success_rate"] - 2 * metrics["violation_rate"]
                payload = {
                    "schema_version": 1, "estimator": estimator.state_dict(),
                    "config": config, "update": update, "metrics": metrics,
                    "teacher_sha256": hashlib.sha256(teacher_path.read_bytes()).hexdigest(),
                    "observation_contract": "rgb_dinov2_spatial_proprio_causal_state_v1",
                }
                atomic_save(payload, output / "latest.pt")
                if score > best_score:
                    best_score = score; atomic_save(payload, output / "best.pt")
                print(json.dumps(metrics), flush=True)
    finally:
        train_env.close(); eval_env.close()


if __name__ == "__main__": main()

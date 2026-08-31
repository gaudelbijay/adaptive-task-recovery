#!/usr/bin/env python3
"""DAgger distillation of the V4 state specialist through spatial DINO features."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

import mani_skill.envs  # noqa: F401
import atr.envs.learned_recovery_v4  # noqa: F401
from probe_v4_temporal_feasibility import preprocess
from train_manipulation_ppo import Agent as StateAgent
from train_v4_permanent_visual_dagger import (
    reconstruct_v4_state_teacher_observation,
)
from train_visual_recovery_dual_teacher_ppo import extract_observation


class SpatialDinoPolicy(nn.Module):
    def __init__(self, proprio_dim: int, action_dim: int):
        super().__init__()
        self.spatial = nn.Sequential(
            nn.Conv2d(384, 32, 1), nn.ReLU(), nn.AdaptiveAvgPool2d((8, 8)),
        )
        self.cls = nn.Sequential(nn.Linear(384, 128), nn.ReLU())
        width = 32 * 8 * 8 + 128 + proprio_dim + 2
        self.actor = nn.Sequential(
            nn.Linear(width, 512), nn.ReLU(),
            nn.Linear(512, 512), nn.ReLU(),
            nn.Linear(512, action_dim),
        )

    def forward(self, features: dict, proprio: torch.Tensor, progress: torch.Tensor):
        patch = features["x_norm_patchtokens"]
        side = int(round(patch.shape[1] ** 0.5))
        patch = patch.transpose(1, 2).reshape(patch.shape[0], 384, side, side)
        spatial = self.spatial(patch).flatten(1)
        cls = self.cls(features["x_norm_clstoken"])
        return torch.tanh(self.actor(torch.cat((spatial, cls, proprio, progress), dim=1)))


def make_env(num_envs: int):
    env = gym.make(
        "LearnedRecovery-v4", num_envs=num_envs, obs_mode="rgb", render_mode=None,
        sim_backend="physx_cuda", control_mode="pd_joint_delta_pos",
        reward_mode="normalized_dense", reconfiguration_freq=1,
        asymmetric_critic_observation=True, vision_camera_size=64,
        required_goals=2, intervention_probability=1.0,
        intervention_types=("permanent_block",), onset_step_range=(0, 0),
        blocker_force=4.0, blocker_return_force=5.0,
        blocker_return_delay_steps=30, terminate_on_violation=True,
        safety_proximity_weight=5.0, constraint_violation_penalty=20.0,
        progress_reward_scale=2.0, completion_bonus=5.0, success_reward=10.0,
    )
    if isinstance(env.action_space, gym.spaces.Dict):
        env = FlattenActionSpaceWrapper(env)
    return ManiSkillVectorEnv(
        env, num_envs, ignore_terminations=True, record_metrics=False,
    )


def actor_inputs(obs):
    rgb, proprio, _ = extract_observation(obs, True, True, False)
    progress = obs["extra"]["critic_goal_resolved"].float()
    return rgb, proprio, progress


@torch.inference_mode()
def evaluate(student, backbone, env, steps):
    obs, _ = env.reset(seed=246_000_000)
    success = torch.zeros(env.num_envs, dtype=torch.bool, device="cuda")
    violation = torch.zeros_like(success)
    for _ in range(steps):
        rgb, proprio, progress = actor_inputs(obs)
        features = backbone.forward_features(preprocess(rgb))
        obs, _, _, _, info = env.step(student(features, proprio, progress))
        success |= info["success"].bool()
        violation |= info["constraint_violated"].bool()
    return {
        "successes": int(success.sum()), "episodes": int(env.num_envs),
        "success_rate": float(success.float().mean()),
        "violations": int(violation.sum()),
        "violation_rate": float(violation.float().mean()),
    }


def atomic_save(payload, path: Path):
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/v4_dino_permanent_student_pilot.json")
    parser.add_argument("--output", default="results/v4_dino_permanent_student_pilot")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    torch.manual_seed(int(config["seed"])); np.random.seed(int(config["seed"]))
    train_env = make_env(int(config["num_envs"]))
    eval_env = make_env(int(config["eval_envs"]))
    teacher_path = Path(config["teacher_checkpoint"])
    teacher_checkpoint = torch.load(teacher_path, map_location="cuda", weights_only=False)
    obs, _ = train_env.reset(seed=int(config["seed"]))
    teacher_obs = reconstruct_v4_state_teacher_observation(obs)
    action_dim = int(np.prod(train_env.single_action_space.shape))
    teacher = StateAgent(teacher_obs.shape[1], action_dim).cuda()
    teacher.load_state_dict(teacher_checkpoint["agent"]); teacher.eval()
    rgb, proprio, progress = actor_inputs(obs)
    backbone = torch.hub.load(
        "facebookresearch/dinov2", "dinov2_vits14", verbose=False,
    ).eval().cuda()
    for parameter in backbone.parameters():
        parameter.requires_grad_(False)
    student = SpatialDinoPolicy(proprio.shape[1], action_dim).cuda()
    optimizer = torch.optim.AdamW(student.parameters(), lr=config["learning_rate"])
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    history = output / "metrics.jsonl"
    best_score = -float("inf")
    try:
        for update in range(1, int(config["updates"]) + 1):
            rgb, proprio, progress = actor_inputs(obs)
            with torch.no_grad():
                features = backbone.forward_features(preprocess(rgb))
                teacher_action = teacher.get_action(
                    reconstruct_v4_state_teacher_observation(obs), deterministic=True,
                ).clamp(-1, 1)
            student_action = student(features, proprio, progress)
            loss = F.mse_loss(student_action, teacher_action)
            optimizer.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(student.parameters(), 1.0); optimizer.step()
            probability = float(config["student_rollout_max"]) * update / int(config["updates"])
            choose_student = torch.rand(int(config["num_envs"]), device="cuda") < probability
            executed = torch.where(choose_student[:, None], student_action.detach(), teacher_action)
            obs, _, _, _, _ = train_env.step(executed)
            if update == 1 or update % int(config["eval_frequency"]) == 0:
                metrics = evaluate(student, backbone, eval_env, int(config["eval_steps"]))
                metrics.update({"update": update, "bc_loss": float(loss.detach())})
                with history.open("a") as handle:
                    handle.write(json.dumps(metrics) + "\n")
                score = metrics["success_rate"] - 2 * metrics["violation_rate"]
                payload = {
                    "schema_version": 1, "student": student.state_dict(),
                    "config": config, "update": update, "metrics": metrics,
                    "teacher_sha256": hashlib.sha256(teacher_path.read_bytes()).hexdigest(),
                    "observation_contract": "rgb_dinov2_spatial_proprio_causal_progress_v1",
                }
                atomic_save(payload, output / "latest.pt")
                if score > best_score:
                    best_score = score; atomic_save(payload, output / "best.pt")
                print(json.dumps(metrics), flush=True)
    finally:
        train_env.close(); eval_env.close()


if __name__ == "__main__":
    main()

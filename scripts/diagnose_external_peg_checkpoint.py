#!/usr/bin/env python3
"""Report geometric competence stages for an official PegInsertion PPO checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

import mani_skill.envs  # noqa: F401
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

from train_manipulation_ppo import Agent


def rate(value: torch.Tensor) -> float:
    return float(value.float().mean())


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=421_500_000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    device = torch.device("cuda")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    task = checkpoint.get("task", {"control_mode": "pd_joint_delta_pos"})
    raw = gym.make(
        "PegInsertionSide-v1", num_envs=args.num_envs, reconfiguration_freq=1,
        obs_mode="state", render_mode=None, sim_backend="physx_cuda",
        control_mode=task["control_mode"], reward_mode="normalized_dense",
    )
    base = raw.unwrapped
    if isinstance(raw.action_space, gym.spaces.Dict):
        raw = FlattenActionSpaceWrapper(raw)
    env = ManiSkillVectorEnv(
        raw, args.num_envs, ignore_terminations=True, record_metrics=False,
    )
    observation, _ = env.reset(seed=args.seed)
    agent = Agent(
        int(np.prod(env.single_observation_space.shape)),
        int(np.prod(env.single_action_space.shape)),
    ).to(device)
    agent.load_state_dict(checkpoint["agent"], strict=True)
    agent.eval()
    action_low = torch.as_tensor(env.single_action_space.low, device=device)
    action_high = torch.as_tensor(env.single_action_space.high, device=device)
    grasped_once = torch.zeros(args.num_envs, dtype=torch.bool, device=device)
    preinserted_once = torch.zeros_like(grasped_once)
    x_aligned_once = torch.zeros_like(grasped_once)
    yz_inside_once = torch.zeros_like(grasped_once)
    success_once = torch.zeros_like(grasped_once)
    minimum_head_yz = torch.full((args.num_envs,), torch.inf, device=device)
    minimum_peg_yz = torch.full_like(minimum_head_yz, torch.inf)
    minimum_hole_distance = torch.full_like(minimum_head_yz, torch.inf)
    maximum_hole_x = torch.full_like(minimum_head_yz, -torch.inf)
    try:
        for _ in range(args.steps):
            action = torch.clamp(agent.get_action(observation, True), action_low, action_high)
            observation, _, _, _, info = env.step(action)
            grasped = base.agent.is_grasping(base.peg, max_angle=20)
            head_goal = (base.goal_pose.inv() * base.peg_head_pose).p
            peg_goal = (base.goal_pose.inv() * base.peg.pose).p
            head_yz = torch.linalg.vector_norm(head_goal[:, 1:], dim=1)
            peg_yz = torch.linalg.vector_norm(peg_goal[:, 1:], dim=1)
            hole_head = info["peg_head_pos_at_hole"]
            radii = base.box_hole_radii
            x_ok = hole_head[:, 0] >= -0.015
            yz_ok = (
                (hole_head[:, 1].abs() <= radii)
                & (hole_head[:, 2].abs() <= radii)
            )
            grasped_once |= grasped
            preinserted_once |= (head_yz < 0.01) & (peg_yz < 0.01)
            x_aligned_once |= x_ok
            yz_inside_once |= yz_ok
            success_once |= info["success"].bool()
            minimum_head_yz = torch.minimum(minimum_head_yz, head_yz)
            minimum_peg_yz = torch.minimum(minimum_peg_yz, peg_yz)
            minimum_hole_distance = torch.minimum(
                minimum_hole_distance,
                torch.linalg.vector_norm(hole_head, dim=1),
            )
            maximum_hole_x = torch.maximum(maximum_hole_x, hole_head[:, 0])
    finally:
        env.close()
    result = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_global_step": int(checkpoint.get("global_step", 0)),
        "episodes": args.num_envs,
        "grasped_once_rate": rate(grasped_once),
        "preinserted_once_rate": rate(preinserted_once),
        "hole_x_threshold_once_rate": rate(x_aligned_once),
        "hole_yz_inside_once_rate": rate(yz_inside_once),
        "success_once_rate": rate(success_once),
        "minimum_head_yz_mean": float(minimum_head_yz.mean()),
        "minimum_peg_yz_mean": float(minimum_peg_yz.mean()),
        "minimum_hole_distance_mean": float(minimum_hole_distance.mean()),
        "maximum_hole_x_mean": float(maximum_hole_x.mean()),
        "maximum_hole_x_q90": float(torch.quantile(maximum_hole_x, 0.9)),
        "mean_insertion_threshold_gap": float(
            (-0.015 - maximum_hole_x).clamp_min(0).mean()
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

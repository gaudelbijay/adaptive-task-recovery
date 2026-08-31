#!/usr/bin/env python3
"""Diagnose V19 controller compatibility with the mechanism-diverse V4 scene.

``normal`` uses V19 exactly as deployed. ``oracle`` replaces only its learned
two-bit progress prediction with evaluator goal-resolution bits while retaining
the same RGB/proprioception encoder and continuous-control actor.  Oracle is an
upper diagnostic, not a deployable method.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

import mani_skill.envs  # noqa: F401
import atr.envs.learned_recovery_v4  # noqa: F401
import atr.envs.learned_recovery_v4_ood  # noqa: F401
from train_visual_recovery_dual_teacher_ppo import (
    VisualAgent, env_kwargs, extract_observation, observation_contract,
    privileged_aux_dim, select_task,
)


SEEDS = (9351, 4796, 1788)
CONDITIONS = (
    "nominal", "ejection", "permanent_block", "temporary_block",
    "reverse_ejection",
)
PROGRESS_SOURCES = ("normal", "oracle", "oracle_defer")


@torch.inference_mode()
def action_with_progress(agent, rgb, proprio, source, resolved):
    latent = agent.encode(rgb)
    if agent.goal_progress_predictor is None:
        raise ValueError("V19 checkpoint lacks its learned progress interface")
    predicted = torch.sigmoid(agent.goal_progress_predictor(latent))
    progress = predicted if source == "normal" else resolved.float()
    mean = agent.actor(torch.cat((latent, proprio, progress), dim=1))
    return torch.tanh(mean), predicted


def run(args, task, seed, condition, progress_source):
    kwargs = env_kwargs(task, evaluation=True)
    kwargs.update({
        "intervention_probability": 0.0 if condition == "nominal" else 1.0,
        "intervention_types": (
            ("ejection",) if condition == "nominal" else (condition,)
        ),
        "onset_step_range": (0, 0), "intervention_force": 6.0,
        "intervention_steps": 24, "blocker_force": 4.0,
        "blocker_return_force": 5.0, "blocker_return_delay_steps": 30,
    })
    if args.env_id == "LearnedRecovery-v4-OOD":
        if not args.visual_domain_profile:
            raise ValueError("OOD evaluation requires --visual-domain-profile")
        kwargs["visual_domain_profile"] = args.visual_domain_profile
    env = gym.make(
        args.env_id, num_envs=args.num_envs,
        reconfiguration_freq=1, max_episode_steps=args.steps, **kwargs,
    )
    if isinstance(env.action_space, gym.spaces.Dict):
        env = FlattenActionSpaceWrapper(env)
    env = ManiSkillVectorEnv(
        env, args.num_envs, ignore_terminations=True, record_metrics=False
    )
    checkpoint_path = Path(args.checkpoint_root) / task["method"] / f"seed_{seed}" / "best.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cuda", weights_only=False)
    if checkpoint.get("observation_contract") != observation_contract(task):
        raise ValueError("checkpoint observation contract mismatch")

    first, _ = env.reset(seed=args.seed_base + seed * 100_000)
    rgb, proprio, critic = extract_observation(
        first, task["asymmetric_critic"], task.get("actor_tcp_pose", False),
        task.get("actor_goal_progress", False),
    )
    action_dim = int(np.prod(env.single_action_space.shape))
    agent = VisualAgent(
        task["image_size"], proprio.shape[1], critic.shape[1], action_dim,
        task["asymmetric_critic"], task.get("augmentation_pad", 0),
        privileged_aux_dim(task), task.get("actor_learned_goal_progress", False),
    ).cuda()
    agent.load_state_dict(checkpoint["agent"])
    agent.eval()

    success_count = violation_count = 0
    progress_correct = progress_total = 0
    try:
        for offset in range(0, args.episodes, args.num_envs):
            obs, _ = env.reset(seed=args.seed_base + seed * 100_000 + offset)
            success = torch.zeros(args.num_envs, dtype=torch.bool, device="cuda")
            violation = torch.zeros_like(success)
            for _ in range(args.steps):
                rgb, proprio, _ = extract_observation(
                    obs, task["asymmetric_critic"], task.get("actor_tcp_pose", False),
                    task.get("actor_goal_progress", False),
                )
                resolved = obs["extra"]["critic_goal_resolved"].bool()
                action, predicted = action_with_progress(
                    agent, rgb, proprio, progress_source, resolved
                )
                if progress_source == "oracle_defer":
                    unwrapped = env.unwrapped
                    waiting = (
                        (unwrapped._intervention_mechanism == 2)
                        & ~unwrapped._temporary_cleared
                    )
                    action = torch.where(waiting[:, None], torch.zeros_like(action), action)
                progress_correct += int(((predicted >= 0.5) == resolved).sum())
                progress_total += int(resolved.numel())
                obs, _, _, _, info = env.step(action)
                success |= info["success"].bool()
                violation |= info["constraint_violated"].bool()
            success_count += int(success.sum())
            violation_count += int(violation.sum())
    finally:
        env.close()
    return {
        "seed": seed, "condition": condition, "progress_source": progress_source,
        "episode_horizon": args.steps,
        "environment": args.env_id,
        "visual_domain_profile": args.visual_domain_profile or "nominal",
        "episodes": args.episodes, "successes": success_count,
        "success_rate": success_count / args.episodes,
        "violations": violation_count, "violation_rate": violation_count / args.episodes,
        "native_progress_bit_accuracy": progress_correct / progress_total,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/visual_recovery_dual_specialist_dagger_v19.json")
    parser.add_argument("--checkpoint-root", default=(
        "results/visual_recovery_ppo/visual_recovery_dual_specialist_dagger_v19"
    ))
    parser.add_argument("--task-index", type=int, required=True)
    parser.add_argument("--episodes", type=int, default=64)
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--steps", type=int, default=240)
    parser.add_argument("--env-id", default="LearnedRecovery-v4")
    parser.add_argument("--visual-domain-profile")
    parser.add_argument("--seed-base", type=int, default=180_000_000)
    parser.add_argument("--output-dir", default="results/v19_on_v4_diagnostic")
    args = parser.parse_args()
    combinations = [
        (seed_index, condition, source)
        for seed_index in range(len(SEEDS))
        for condition in CONDITIONS
        for source in PROGRESS_SOURCES
    ]
    if not 0 <= args.task_index < len(combinations):
        raise ValueError(f"task-index must be in [0, {len(combinations) - 1}]")
    seed_index, condition, source = combinations[args.task_index]
    config = json.loads(Path(args.config).read_text())
    task, _ = select_task(config, seed_index)
    seed = int(task["seed"])
    if seed != SEEDS[seed_index]:
        raise ValueError("config seed order differs from the frozen diagnostic order")
    result = run(args, task, seed, condition, source)
    output = Path(args.output_dir) / f"seed_{seed}_{condition}_{source}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

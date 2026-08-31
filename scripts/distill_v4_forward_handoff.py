#!/usr/bin/env python3
"""Distill the legacy forward expert into the V4 handoff state contract.

The teacher receives only its historical V3 state projection.  The student
receives the same V4 state projection used by the learned-router evaluator.
Mechanism IDs and intervention targets are never policy inputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

import mani_skill.envs  # noqa: F401
import atr.envs.learned_recovery_v4  # noqa: F401
from train_manipulation_ppo import Agent
from train_v4_permanent_visual_dagger import reconstruct_v4_state_teacher_observation
from train_visual_recovery_dual_teacher_ppo import reconstruct_state_teacher_observation


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_env(num_envs: int):
    env = gym.make(
        "LearnedRecovery-v4", num_envs=num_envs, obs_mode="state_dict",
        render_mode=None, sim_backend="physx_cuda", control_mode="pd_joint_delta_pos",
        reward_mode="normalized_dense", reconfiguration_freq=1,
        asymmetric_critic_observation=True, required_goals=2,
        intervention_probability=1.0, intervention_types=("ejection",),
        onset_step_range=(0, 0), intervention_force=6.0, intervention_steps=24,
        terminate_on_violation=True, safety_proximity_weight=5.0,
        constraint_violation_penalty=20.0, progress_reward_scale=2.0,
        completion_bonus=5.0, success_reward=10.0,
    )
    if isinstance(env.action_space, gym.spaces.Dict):
        env = FlattenActionSpaceWrapper(env)
    return ManiSkillVectorEnv(env, num_envs, ignore_terminations=True, record_metrics=False)


@torch.inference_mode()
def evaluate(student, args, seed: int) -> dict:
    env = make_env(args.eval_envs)
    try:
        obs, _ = env.reset(seed=seed)
        success = torch.zeros(args.eval_envs, dtype=torch.bool, device="cuda")
        violation = torch.zeros_like(success)
        for step in range(1, args.horizon + 1):
            if step < args.handoff_step:
                action = torch.zeros((args.eval_envs,) + env.single_action_space.shape, device="cuda")
            else:
                action = student.get_action(
                    reconstruct_v4_state_teacher_observation(obs), deterministic=True,
                ).clamp(-1, 1)
            obs, _, _, _, info = env.step(action)
            success |= info["success"].bool()
            violation |= info["constraint_violated"].bool()
        return {
            "episodes": args.eval_envs,
            "successes": int(success.sum()),
            "safe_successes": int((success & ~violation).sum()),
            "violations": int(violation.sum()),
        }
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=9351)
    parser.add_argument("--num-envs", type=int, default=512)
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--teacher-rollout-episodes", type=int, default=2)
    parser.add_argument("--horizon", type=int, default=240)
    parser.add_argument("--handoff-step", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--eval-envs", type=int, default=256)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("distillation requires CUDA")
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    teacher_path = Path(args.teacher)
    teacher_payload = torch.load(teacher_path, map_location="cuda", weights_only=False)
    env = make_env(args.num_envs)
    try:
        obs, _ = env.reset(seed=args.seed)
        v3 = reconstruct_state_teacher_observation(obs)
        v4 = reconstruct_v4_state_teacher_observation(obs)
        action_dim = int(np.prod(env.single_action_space.shape))
        teacher = Agent(v3.shape[1], action_dim).cuda()
        teacher.load_state_dict(teacher_payload["agent"], strict=True); teacher.eval()
        student = Agent(v4.shape[1], action_dim).cuda()
        optimizer = torch.optim.AdamW(student.actor_mean.parameters(), lr=args.learning_rate, weight_decay=1e-5)
        losses = []
        for episode in range(args.episodes):
            obs, _ = env.reset(seed=args.seed + episode * 100_000)
            for step in range(1, args.horizon + 1):
                v3 = reconstruct_state_teacher_observation(obs)
                v4 = reconstruct_v4_state_teacher_observation(obs)
                with torch.no_grad():
                    target = teacher.get_action(v3, deterministic=True).clamp(-1, 1)
                prediction = student.actor_mean(v4)
                loss = F.mse_loss(prediction, target)
                optimizer.zero_grad(set_to_none=True); loss.backward()
                torch.nn.utils.clip_grad_norm_(student.actor_mean.parameters(), 5.0)
                optimizer.step(); losses.append(float(loss.detach()))
                if step < args.handoff_step:
                    action = torch.zeros_like(target)
                elif episode < args.teacher_rollout_episodes:
                    action = target
                else:
                    # DAgger: visit the student's own state distribution while
                    # continuing to relabel every visited state with the teacher.
                    action = prediction.detach().clamp(-1, 1)
                obs, _, _, _, _ = env.step(action)
    finally:
        env.close()
    metrics = evaluate(student, args, args.seed + 90_000_000)
    task = {
        "method": "forward_ejection_v3_to_v4_handoff_distillation",
        "env_id": "LearnedRecovery-v4", "control_mode": "pd_joint_delta_pos",
        "seed": args.seed, "handoff_step": args.handoff_step,
    }
    payload = {
        "schema_version": 1, "task": task, "agent": student.state_dict(),
        "global_step": args.episodes * args.horizon * args.num_envs,
        "teacher_checkpoint": str(teacher_path),
        "teacher_sha256": sha256(teacher_path),
        "distillation_mse_last_100": float(np.mean(losses[-100:])),
        "evaluation": metrics,
        "forbidden_runtime_inputs": [
            "mechanism ID", "intervention target", "critic_goal_resolved", "future observation",
        ],
    }
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)
    manifest = output.with_suffix(".json")
    manifest.write_text(json.dumps({**payload, "agent": "stored in checkpoint"}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({**metrics, "distillation_mse_last_100": payload["distillation_mse_last_100"]}, indent=2))


if __name__ == "__main__":
    main()

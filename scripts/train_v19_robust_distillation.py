#!/usr/bin/env python3
"""Post-hoc robust RGB self-distillation from a frozen V19 policy."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import random
from pathlib import Path

import gymnasium as gym
import mani_skill.envs  # noqa: F401
import numpy as np
import torch
import torch.nn.functional as F
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

from train_visual_recovery_dual_teacher_ppo import (
    RandomShiftsAug,
    VisualAgent,
    env_kwargs,
    extract_observation,
    file_sha256,
    observation_contract,
    privileged_aux_dim,
    select_task,
    visual_progress_target,
)


def robust_augment(
    rgb: torch.Tensor,
    *,
    pad: int,
    brightness: tuple[float, float],
    channel_gain: tuple[float, float],
    probability: float,
) -> torch.Tensor:
    """Return float NCHW images with per-sample spatial/appearance jitter."""
    if not 0.0 <= probability <= 1.0:
        raise ValueError("augmentation probability must lie in [0, 1]")
    if brightness[0] <= 0 or brightness[1] < brightness[0]:
        raise ValueError("invalid brightness range")
    if channel_gain[0] <= 0 or channel_gain[1] < channel_gain[0]:
        raise ValueError("invalid channel-gain range")
    image = rgb.permute(0, 3, 1, 2).float().div(255.0)
    augmented = RandomShiftsAug(pad)(image)
    count = image.shape[0]
    brightness_scale = torch.empty(
        (count, 1, 1, 1), device=image.device, dtype=image.dtype,
    ).uniform_(*brightness)
    channel_scale = torch.empty(
        (count, 3, 1, 1), device=image.device, dtype=image.dtype,
    ).uniform_(*channel_gain)
    augmented = (augmented * brightness_scale * channel_scale).clamp(0.0, 1.0)
    mask = torch.rand((count, 1, 1, 1), device=image.device) < probability
    return torch.where(mask, augmented, image)


def action_from_latent(agent: VisualAgent, latent: torch.Tensor, proprio: torch.Tensor):
    parts = [latent, proprio]
    if agent.goal_progress_predictor is not None:
        parts.append(torch.sigmoid(agent.goal_progress_predictor(latent)))
    return torch.tanh(agent.actor(torch.cat(parts, dim=1)))


def atomic_save(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", default="results/visual_recovery_ppo")
    parser.add_argument(
        "--task-index", type=int,
        default=int(os.environ.get("SLURM_ARRAY_TASK_ID", "0")),
    )
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    config_path = Path(args.config)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    task, task_count = select_task(config, args.task_index)
    if args.preflight:
        print(json.dumps({"task_count": task_count, **task}, indent=2))
        return
    if not torch.cuda.is_available():
        raise RuntimeError("robust visual distillation requires CUDA")
    if task.get("registration_module"):
        importlib.import_module(task["registration_module"])

    seed = int(task["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda")
    run_dir = Path(args.output) / config["name"] / task["method"] / f"seed_{seed}"
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite robust-distillation run: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "task.json").write_text(json.dumps(task, indent=2) + "\n")

    envs = gym.make(task["env_id"], num_envs=task["num_envs"], **env_kwargs(task))
    if isinstance(envs.action_space, gym.spaces.Dict):
        envs = FlattenActionSpaceWrapper(envs)
    envs = ManiSkillVectorEnv(envs, task["num_envs"], record_metrics=True)
    observation, _ = envs.reset(seed=seed)
    rgb, proprio, critic = extract_observation(
        observation, task["asymmetric_critic"], task.get("actor_tcp_pose", False),
        task.get("actor_goal_progress", False),
    )
    action_dim = int(np.prod(envs.single_action_space.shape))
    agent = VisualAgent(
        task["image_size"], proprio.shape[1], critic.shape[1], action_dim,
        task["asymmetric_critic"], 0, privileged_aux_dim(task),
        task.get("actor_learned_goal_progress", False),
    ).to(device)
    teacher = VisualAgent(
        task["image_size"], proprio.shape[1], critic.shape[1], action_dim,
        task["asymmetric_critic"], 0, privileged_aux_dim(task),
        task.get("actor_learned_goal_progress", False),
    ).to(device)
    checkpoint_path = Path(str(task["distillation_teacher_checkpoint"]).format(seed=seed))
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    expected_contract = observation_contract(task)
    if checkpoint.get("observation_contract") != expected_contract:
        raise ValueError("V19 teacher observation contract mismatch")
    agent.load_state_dict(checkpoint["agent"], strict=True)
    teacher.load_state_dict(checkpoint["agent"], strict=True)
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.Adam(agent.parameters(), lr=float(config["learning_rate"]), eps=1e-5)

    aug = task["robust_augmentation"]
    pad = int(aug["pad"])
    brightness = tuple(float(value) for value in aug["brightness_range"])
    channel_gain = tuple(float(value) for value in aug["channel_gain_range"])
    probability = float(aug["probability"])
    updates = int(task["distillation_updates"])
    rollout_max = float(task["student_rollout_max"])
    loss_history = []
    original_action_history = []
    augmented_action_history = []
    progress_history = []
    student_fraction_history = []
    agent.train()
    for update in range(updates):
        rgb, proprio, _ = extract_observation(
            observation, task["asymmetric_critic"], task.get("actor_tcp_pose", False),
            task.get("actor_goal_progress", False),
        )
        with torch.no_grad():
            target_action = teacher.get_action(rgb, proprio, deterministic=True)
            progress_target = visual_progress_target(observation)
        original_latent = agent.encode(rgb)
        original_action = action_from_latent(agent, original_latent, proprio)
        augmented_image = robust_augment(
            rgb, pad=pad, brightness=brightness, channel_gain=channel_gain,
            probability=probability,
        )
        augmented_latent = agent.encoder(augmented_image)
        augmented_action = action_from_latent(agent, augmented_latent, proprio)
        original_action_loss = F.mse_loss(original_action, target_action)
        augmented_action_loss = F.mse_loss(augmented_action, target_action)
        progress_loss = original_action.new_zeros(())
        if agent.goal_progress_predictor is not None:
            progress_loss = 0.5 * (
                F.binary_cross_entropy_with_logits(
                    agent.goal_progress_predictor(original_latent), progress_target,
                )
                + F.binary_cross_entropy_with_logits(
                    agent.goal_progress_predictor(augmented_latent), progress_target,
                )
            )
        consistency_loss = 1.0 - F.cosine_similarity(
            augmented_latent, original_latent.detach(), dim=1,
        ).mean()
        loss = (
            float(task["original_action_weight"]) * original_action_loss
            + float(task["augmented_action_weight"]) * augmented_action_loss
            + float(task["progress_weight"]) * progress_loss
            + float(task["latent_consistency_weight"]) * consistency_loss
        )
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
        optimizer.step()

        with torch.no_grad():
            student_probability = rollout_max * update / max(updates - 1, 1)
            student_mask = torch.rand(
                (target_action.shape[0], 1), device=device,
            ) < student_probability
            executed = torch.where(student_mask, original_action.detach(), target_action)
            observation, _, _, _, _ = envs.step(executed)
        loss_history.append(float(loss.detach()))
        original_action_history.append(float(original_action_loss.detach()))
        augmented_action_history.append(float(augmented_action_loss.detach()))
        progress_history.append(float(progress_loss.detach()))
        student_fraction_history.append(float(student_mask.float().mean()))
        if (update + 1) % int(config["log_freq"]) == 0:
            with (run_dir / "metrics.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({
                    "update": update + 1,
                    "global_step": (update + 1) * int(task["num_envs"]),
                    "loss": float(np.mean(loss_history[-config["log_freq"]:])),
                    "original_action_loss": float(np.mean(
                        original_action_history[-config["log_freq"]:]
                    )),
                    "augmented_action_loss": float(np.mean(
                        augmented_action_history[-config["log_freq"]:]
                    )),
                    "progress_loss": float(np.mean(progress_history[-config["log_freq"]:])),
                }) + "\n")

    envs.close()
    global_step = updates * int(task["num_envs"])
    source_hashes = {
        "trainer": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "base_visual_trainer": hashlib.sha256(
            Path(__file__).with_name("train_visual_recovery_dual_teacher_ppo.py").read_bytes()
        ).hexdigest(),
    }
    payload = {
        "schema_version": 1,
        "observation_contract": expected_contract,
        "source_sha256": source_hashes,
        "task": task,
        "agent": agent.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iteration": updates,
        "global_step": global_step,
        "best_score": -float(np.mean(loss_history[-100:])),
        "best_metrics": {
            "mean_last_100_loss": float(np.mean(loss_history[-100:])),
            "mean_last_100_original_action_loss": float(np.mean(original_action_history[-100:])),
            "mean_last_100_augmented_action_loss": float(np.mean(augmented_action_history[-100:])),
            "mean_last_100_progress_loss": float(np.mean(progress_history[-100:])),
        },
    }
    atomic_save(payload, run_dir / "best.pt")
    atomic_save(payload, run_dir / "latest.pt")
    completion = {
        "global_step": global_step,
        "distillation_updates": updates,
        "distillation_transitions": global_step,
        "teacher_checkpoint": str(checkpoint_path),
        "teacher_checkpoint_sha256": file_sha256(checkpoint_path),
        "mean_student_rollout_fraction": float(np.mean(student_fraction_history)),
        **payload["best_metrics"],
        "source_sha256": source_hashes,
    }
    (run_dir / "TRAINING_COMPLETE.json").write_text(
        json.dumps(completion, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(completion, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Checkpointed RGB PPO for LearnedRecovery-v1.

The deployed actor is deliberately restricted to RGB, robot qpos/qvel, and
the parsed two-token instruction.  An experiment may use simulator state in a
training-only asymmetric critic.  Optional action-conditioned temporal latent
prediction supplies a self-supervised representation loss.  Actions use a
tanh-transformed Gaussian, so PPO scores exactly the bounded command executed
by ManiSkill instead of scoring an unexecuted, subsequently clipped sample.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import random
import signal
import time
from collections import defaultdict
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

import mani_skill.envs  # noqa: F401
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv


def layer_init(layer, std=np.sqrt(2), bias=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias)
    return layer


class RandomShiftsAug(nn.Module):
    """Integer translation augmentation used by DrQ-v2."""

    def __init__(self, pad=4):
        super().__init__()
        self.pad = int(pad)

    def forward(self, x):
        if self.pad == 0:
            return x
        n, _, h, w = x.shape
        if h != w:
            raise ValueError("random-shift augmentation expects square images")
        x = F.pad(x, (self.pad,) * 4, mode="replicate")
        eps = 1.0 / (h + 2 * self.pad)
        coordinates = torch.linspace(
            -1.0 + eps, 1.0 - eps, h + 2 * self.pad,
            device=x.device, dtype=x.dtype,
        )[:h]
        grid_y, grid_x = torch.meshgrid(coordinates, coordinates, indexing="ij")
        base = torch.stack((grid_x, grid_y), dim=-1).unsqueeze(0).repeat(n, 1, 1, 1)
        shift = torch.randint(0, 2 * self.pad + 1, (n, 1, 1, 2), device=x.device)
        shift = shift.to(x.dtype) * (2.0 / (h + 2 * self.pad))
        return F.grid_sample(x, base + shift, padding_mode="zeros", align_corners=False)


class VisualAgent(nn.Module):
    def __init__(self, image_size, proprio_dim, critic_dim, action_dim, asymmetric, aug_pad):
        super().__init__()
        self.asymmetric = bool(asymmetric)
        self.augmentation = RandomShiftsAug(aug_pad)
        conv = nn.Sequential(
            nn.Conv2d(3, 32, 8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1), nn.ReLU(), nn.Flatten(),
        )
        with torch.no_grad():
            flat_dim = conv(torch.zeros(1, 3, image_size, image_size)).shape[1]
        self.encoder = nn.Sequential(conv, layer_init(nn.Linear(flat_dim, 256)), nn.ReLU())
        self.actor = nn.Sequential(
            layer_init(nn.Linear(256 + proprio_dim, 512)), nn.ReLU(),
            layer_init(nn.Linear(512, action_dim), std=0.01 * np.sqrt(2)),
        )
        value_input = critic_dim if self.asymmetric else 256 + proprio_dim
        self.critic = nn.Sequential(
            layer_init(nn.Linear(value_input, 512)), nn.ReLU(),
            layer_init(nn.Linear(512, 1), std=1.0),
        )
        self.actor_logstd = nn.Parameter(torch.full((1, action_dim), -0.5))
        self.temporal_predictor = nn.Sequential(
            layer_init(nn.Linear(256 + action_dim, 512)), nn.ReLU(),
            layer_init(nn.Linear(512, 256), std=1.0),
        )

    def encode(self, rgb, augment=False):
        image = rgb.permute(0, 3, 1, 2).float().div(255.0)
        if augment:
            image = self.augmentation(image)
        return self.encoder(image)

    def features(self, rgb, proprio, critic_state, augment=False):
        latent = self.encode(rgb, augment=augment)
        actor_features = torch.cat((latent, proprio), dim=1)
        value_features = critic_state if self.asymmetric else actor_features
        return latent, actor_features, value_features

    def get_value(self, rgb, proprio, critic_state):
        _, _, value_features = self.features(rgb, proprio, critic_state)
        return self.critic(value_features)

    def get_action(self, rgb, proprio, deterministic=False):
        latent = self.encode(rgb)
        mean = self.actor(torch.cat((latent, proprio), dim=1))
        if deterministic:
            return torch.tanh(mean)
        return torch.tanh(Normal(mean, self.actor_logstd.exp().expand_as(mean)).sample())

    def action_and_value(
        self, rgb, proprio, critic_state, pre_tanh_action=None, augment=False,
    ):
        latent, actor_features, value_features = self.features(
            rgb, proprio, critic_state, augment=augment,
        )
        mean = self.actor(actor_features)
        distribution = Normal(mean, self.actor_logstd.exp().expand_as(mean))
        if pre_tanh_action is None:
            pre_tanh_action = distribution.sample()
        action = torch.tanh(pre_tanh_action)
        logprob = (
            distribution.log_prob(pre_tanh_action)
            - torch.log(1.0 - action.square() + 1e-6)
        ).sum(1)
        return pre_tanh_action, action, logprob, -logprob, self.critic(value_features), latent


ACTOR_EXTRA_KEYS = ("instruction",)
CRITIC_EXTRA_KEYS = (
    "tcp_pose", "instruction", "goal_progress",
    "critic_red_cube_pose", "critic_blue_cube_pose",
    "critic_red_goal_pos", "critic_blue_goal_pos",
    "critic_red_sweeper_pose", "critic_blue_sweeper_pose",
    "critic_protected_pose",
)


def extract_observation(obs, asymmetric):
    """Return structurally separated actor and critic inputs from raw RGB obs."""
    rgb = obs["sensor_data"]["base_camera"]["rgb"]
    actor_parts = [obs["agent"]["qpos"], obs["agent"]["qvel"]]
    actor_parts.extend(obs["extra"][key] for key in ACTOR_EXTRA_KEYS)
    proprio = torch.cat(actor_parts, dim=1)
    if asymmetric:
        missing = [key for key in CRITIC_EXTRA_KEYS if key not in obs["extra"]]
        if missing:
            raise KeyError(f"asymmetric critic fields missing: {missing}")
        critic = torch.cat(
            [obs["agent"]["qpos"], obs["agent"]["qvel"]]
            + [obs["extra"][key] for key in CRITIC_EXTRA_KEYS], dim=1,
        )
    else:
        critic = proprio.new_zeros((proprio.shape[0], 0))
    return rgb, proprio, critic


def atomic_save(payload, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def rng_state():
    return {
        "python": random.getstate(), "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([item.cpu() for item in state["cuda"]])


def select_task(config, index):
    tasks = [{**experiment, "seed": seed} for experiment in config["experiments"] for seed in config["seeds"]]
    if not 0 <= index < len(tasks):
        raise ValueError(f"task-index must be in [0, {len(tasks) - 1}]")
    return tasks[index], len(tasks)


def metric_success(metrics):
    for key in ("success_once", "success_at_end", "success"):
        if key in metrics:
            return float(metrics[key])
    return float("nan")


def env_kwargs(task, evaluation=False):
    kwargs = {
        "obs_mode": "rgb", "render_mode": None, "sim_backend": "physx_cuda",
        "control_mode": task["control_mode"],
        "reward_mode": task.get("reward_mode", "normalized_dense"),
        "vision_camera_size": int(task.get("image_size", 64)),
        "asymmetric_critic_observation": bool(task.get("asymmetric_critic", False)),
    }
    kwargs.update(task.get("env_kwargs", {}))
    if evaluation:
        kwargs.update(task.get("eval_env_kwargs", {}))
    return kwargs


def checkpoint_payload(agent, optimizer, iteration, global_step, best_score, best_metrics, task):
    return {
        "schema_version": 1, "observation_contract": "rgb_qpos_qvel_instruction_v1",
        "task": task, "agent": agent.state_dict(), "optimizer": optimizer.state_dict(),
        "iteration": iteration, "global_step": global_step,
        "best_score": best_score, "best_metrics": best_metrics, "rng": rng_state(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/visual_recovery_ppo_gate_v1.json")
    parser.add_argument("--output", default="results/visual_recovery_ppo")
    parser.add_argument("--task-index", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", "0")))
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    task, task_count = select_task(config, args.task_index)
    if args.preflight:
        print(json.dumps({"task_count": task_count, "task_index": args.task_index, **task}, indent=2))
        return
    if not torch.cuda.is_available():
        raise RuntimeError("visual PPO requires CUDA")
    if task.get("registration_module"):
        importlib.import_module(task["registration_module"])

    seed = int(task["seed"])
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda")
    run_dir = Path(args.output) / config["name"] / task["method"] / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "task.json").write_text(json.dumps(task, indent=2) + "\n", encoding="utf-8")

    envs = gym.make(task["env_id"], num_envs=task["num_envs"], **env_kwargs(task))
    eval_envs = gym.make(
        task["env_id"], num_envs=config["num_eval_envs"], reconfiguration_freq=1,
        **env_kwargs(task, evaluation=True),
    )
    if isinstance(envs.action_space, gym.spaces.Dict):
        envs = FlattenActionSpaceWrapper(envs); eval_envs = FlattenActionSpaceWrapper(eval_envs)
    envs = ManiSkillVectorEnv(envs, task["num_envs"], record_metrics=True)
    eval_envs = ManiSkillVectorEnv(eval_envs, config["num_eval_envs"], ignore_terminations=True, record_metrics=True)
    initial_obs, _ = envs.reset(seed=seed)
    initial_rgb, initial_proprio, initial_critic = extract_observation(initial_obs, task["asymmetric_critic"])
    action_dim = int(np.prod(envs.single_action_space.shape))
    agent = VisualAgent(
        task["image_size"], initial_proprio.shape[1], initial_critic.shape[1], action_dim,
        task["asymmetric_critic"], task.get("augmentation_pad", 0),
    ).to(device)
    optimizer = torch.optim.Adam(agent.parameters(), lr=config["learning_rate"], eps=1e-5)

    latest_path, best_path = run_dir / "latest.pt", run_dir / "best.pt"
    start_iteration, global_step, best_score, best_metrics = 1, 0, float("-inf"), {}
    if latest_path.exists():
        checkpoint = torch.load(latest_path, map_location=device, weights_only=False)
        if checkpoint["task"] != task:
            raise ValueError("checkpoint task does not match immutable task configuration")
        agent.load_state_dict(checkpoint["agent"]); optimizer.load_state_dict(checkpoint["optimizer"])
        start_iteration = int(checkpoint["iteration"]) + 1
        global_step, best_score = int(checkpoint["global_step"]), float(checkpoint["best_score"])
        best_metrics = dict(checkpoint["best_metrics"]); restore_rng(checkpoint["rng"])
    elif task.get("init_checkpoint"):
        initialization_path = Path(str(task["init_checkpoint"]).format(seed=seed))
        if not initialization_path.exists():
            raise FileNotFoundError(f"initialization checkpoint unavailable: {initialization_path}")
        initialization = torch.load(initialization_path, map_location=device, weights_only=False)
        if initialization.get("observation_contract") != "rgb_qpos_qvel_instruction_v1":
            raise ValueError("initialization checkpoint has an incompatible observation contract")
        agent.load_state_dict(initialization["agent"], strict=True)
        (run_dir / "initialization.json").write_text(json.dumps({
            "checkpoint": str(initialization_path),
            "source_task": initialization["task"],
            "source_iteration": int(initialization["iteration"]),
            "source_global_step": int(initialization["global_step"]),
        }, indent=2) + "\n", encoding="utf-8")

    n, t = int(task["num_envs"]), int(task["num_steps"])
    batch_size = n * t
    iterations = int(task["total_timesteps"]) // batch_size
    minibatch_size = batch_size // int(config["num_minibatches"])
    h = w = int(task["image_size"])
    pdim, cdim = initial_proprio.shape[1], initial_critic.shape[1]
    rgbs = torch.empty((t, n, h, w, 3), dtype=torch.uint8, device=device)
    next_rgbs = torch.empty_like(rgbs)
    proprios = torch.empty((t, n, pdim), device=device)
    critic_states = torch.empty((t, n, cdim), device=device)
    pre_actions = torch.empty((t, n, action_dim), device=device)
    logprobs = torch.empty((t, n), device=device)
    rewards = torch.empty((t, n), device=device)
    dones = torch.empty((t, n), device=device)
    next_dones = torch.empty((t, n), device=device)
    values = torch.empty((t, n), device=device)
    next_obs = initial_obs
    next_done = torch.zeros(n, device=device)
    stop_requested = False

    def request_stop(_signum, _frame):
        nonlocal stop_requested
        stop_requested = True
    signal.signal(signal.SIGUSR1, request_stop)
    history_path = run_dir / "metrics.jsonl"
    started = time.time()

    for iteration in range(start_iteration, iterations + 1):
        if config.get("anneal_lr", True):
            fraction = 1.0 - (iteration - 1.0) / iterations
            optimizer.param_groups[0]["lr"] = fraction * config["learning_rate"]

        if iteration == start_iteration or iteration % int(config["eval_freq"]) == 0:
            eval_obs, _ = eval_envs.reset(seed=seed + 10_000 + iteration)
            eval_metrics = defaultdict(list)
            eval_maxima = {
                key: torch.zeros(config["num_eval_envs"], device=device)
                for key in ("goals_completed", "goals_unavailable", "constraint_violated")
            }
            with torch.no_grad():
                for _ in range(int(task["num_eval_steps"])):
                    ergb, eprop, _ = extract_observation(eval_obs, task["asymmetric_critic"])
                    eval_obs, _, _, _, info = eval_envs.step(agent.get_action(ergb, eprop, True))
                    for key in eval_maxima:
                        if key in info:
                            eval_maxima[key] = torch.maximum(
                                eval_maxima[key], info[key].detach().float().reshape(-1),
                            )
                    if "final_info" in info:
                        mask = info["_final_info"]
                        for key, value in info["final_info"]["episode"].items():
                            eval_metrics[key].append(value[mask].float())
            means = {key: float(torch.cat(value).mean()) for key, value in eval_metrics.items() if value and torch.cat(value).numel()}
            means.update({key: float(value.mean()) for key, value in eval_maxima.items()})
            success = metric_success(means)
            failure = float(means.get("constraint_violated", means.get("fail_once", 0.0)))
            # Return breaks exact success/safety ties without ever outweighing
            # a single percentage point of the primary metric.
            score = (
                success - float(config.get("selection_failure_penalty", 0.0)) * failure
                + 1e-6 * float(means.get("return", 0.0))
            )
            with history_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"iteration": iteration, "global_step": global_step, "eval": means, "elapsed_seconds": time.time() - started}) + "\n")
            if np.isfinite(score) and score > best_score:
                best_score, best_metrics = score, means
                atomic_save(checkpoint_payload(agent, optimizer, iteration - 1, global_step, best_score, best_metrics, task), best_path)

        final_values = torch.zeros((t, n), device=device)
        agent.eval()
        for step in range(t):
            global_step += n
            rgb, proprio, critic_state = extract_observation(next_obs, task["asymmetric_critic"])
            rgbs[step], proprios[step], critic_states[step], dones[step] = rgb, proprio, critic_state, next_done
            with torch.no_grad():
                pre, action, logprob, _, value, _ = agent.action_and_value(rgb, proprio, critic_state)
            pre_actions[step], logprobs[step], values[step] = pre, logprob, value.flatten()
            next_obs, reward, terminated, truncated, info = envs.step(action)
            next_done = (terminated | truncated).float()
            next_dones[step] = next_done
            next_rgb, _, _ = extract_observation(next_obs, task["asymmetric_critic"])
            next_rgbs[step] = next_rgb
            rewards[step] = reward.view(-1)
            if "final_info" in info:
                mask = info["_final_info"]
                frgb, fprop, fcritic = extract_observation(info["final_observation"], task["asymmetric_critic"])
                with torch.no_grad():
                    final_values[step, mask] = agent.get_value(frgb[mask], fprop[mask], fcritic[mask]).view(-1)

        with torch.no_grad():
            nrgb, nprop, ncritic = extract_observation(next_obs, task["asymmetric_critic"])
            next_value = agent.get_value(nrgb, nprop, ncritic).reshape(1, -1)
            advantages = torch.zeros_like(rewards)
            last_gae = 0
            for step in reversed(range(t)):
                if step == t - 1:
                    next_not_done, following_value = 1.0 - next_done, next_value
                else:
                    next_not_done, following_value = 1.0 - dones[step + 1], values[step + 1]
                real_next = next_not_done * following_value + final_values[step]
                delta = rewards[step] + config["gamma"] * real_next - values[step]
                advantages[step] = last_gae = delta + config["gamma"] * config["gae_lambda"] * next_not_done * last_gae
            returns = advantages + values

        flat = lambda x: x.reshape((-1,) + x.shape[2:])
        b_rgb, b_next_rgb = flat(rgbs), flat(next_rgbs)
        b_prop = flat(proprios)
        # A symmetric critic has an intentional zero-width privileged tensor;
        # spell out both dimensions because reshape cannot infer ``-1`` from
        # an empty tensor.
        b_critic = critic_states.reshape(batch_size, cdim)
        b_pre = flat(pre_actions)
        b_logprob, b_adv, b_return = logprobs.reshape(-1), advantages.reshape(-1), returns.reshape(-1)
        b_nonterminal = (1.0 - next_dones).reshape(-1)
        indices = np.arange(batch_size)
        agent.train()
        loss_metrics = defaultdict(float)
        updates = 0
        for _ in range(int(config["update_epochs"])):
            np.random.shuffle(indices)
            for start in range(0, batch_size, minibatch_size):
                mb = indices[start:start + minibatch_size]
                _, action, new_logprob, entropy, new_value, latent = agent.action_and_value(
                    b_rgb[mb], b_prop[mb], b_critic[mb], b_pre[mb], augment=True,
                )
                ratio = (new_logprob - b_logprob[mb]).exp()
                adv = b_adv[mb]
                adv = (adv - adv.mean()) / (adv.std() + 1e-8)
                pg_loss = torch.maximum(-adv * ratio, -adv * torch.clamp(ratio, 0.8, 1.2)).mean()
                value_loss = 0.5 * (new_value.view(-1) - b_return[mb]).square().mean()
                temporal_loss = latent.new_zeros(())
                temporal_coefficient = float(task.get("temporal_ssl_coefficient", 0.0))
                if temporal_coefficient:
                    with torch.no_grad():
                        target = F.normalize(agent.encode(b_next_rgb[mb], augment=True), dim=1)
                    prediction = F.normalize(agent.temporal_predictor(torch.cat((latent, action), dim=1)), dim=1)
                    weights = b_nonterminal[mb]
                    temporal_loss = ((prediction - target).square().mean(1) * weights).sum() / weights.sum().clamp_min(1.0)
                loss = pg_loss + config["value_coefficient"] * value_loss - config.get("entropy_coefficient", 0.0) * entropy.mean() + temporal_coefficient * temporal_loss
                optimizer.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), 0.5); optimizer.step()
                for key, value in (("policy", pg_loss), ("value", value_loss), ("temporal", temporal_loss)):
                    loss_metrics[key] += float(value.detach())
                updates += 1

        if iteration % int(config.get("log_freq", 10)) == 0:
            with history_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"iteration": iteration, "global_step": global_step, "train_loss": {key: value / updates for key, value in loss_metrics.items()}, "steps_per_second": global_step / max(time.time() - started, 1e-6)}) + "\n")
        if iteration % int(config["checkpoint_freq"]) == 0 or iteration == iterations or stop_requested:
            atomic_save(checkpoint_payload(agent, optimizer, iteration, global_step, best_score, best_metrics, task), latest_path)
        if stop_requested:
            break

    envs.close(); eval_envs.close()
    completed = global_step >= iterations * batch_size
    if completed:
        (run_dir / "TRAINING_COMPLETE.json").write_text(json.dumps({"global_step": global_step, "best_score": best_score, "best_metrics": best_metrics}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"method": task["method"], "seed": seed, "global_step": global_step, "best_score": best_score, "best_metrics": best_metrics, "completed": completed}, indent=2))


if __name__ == "__main__":
    main()

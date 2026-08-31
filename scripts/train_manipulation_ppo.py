#!/usr/bin/env python3
"""Checkpointed state-based PPO for established ManiSkill manipulation tasks.

The network, Gaussian initialization, and PPO loss follow ManiSkill v3.0.0b22's
official readable ``ppo.py``; 50M budgets and task-specific parallelization
follow its official ``baselines.sh``/``ppo_fast.py`` commands. This version
does not use the compiled CUDA-graph optimization. It adds exact optimizer,
counter, and RNG resume; atomic latest/best checkpoints; and immutable JSON
experiment selection for Jarvis arrays. Environment streams are re-seeded on
continuation because SAPIEN does not expose a portable serialized simulator/RNG
snapshot; model, optimizer, counters, and Python/NumPy/Torch RNG states resume.
It trains real continuous control—no ATR teleport-on-success code is imported
or called.
"""

from __future__ import annotations

import argparse
import hashlib
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
from torch.distributions.normal import Normal

import mani_skill.envs  # noqa: F401
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv


def _layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, observation_dim: int, action_dim: int):
        super().__init__()
        def network(output_dim, output_std=np.sqrt(2)):
            return nn.Sequential(
                _layer_init(nn.Linear(observation_dim, 256)), nn.Tanh(),
                _layer_init(nn.Linear(256, 256)), nn.Tanh(),
                _layer_init(nn.Linear(256, 256)), nn.Tanh(),
                _layer_init(nn.Linear(256, output_dim), std=output_std),
            )
        self.critic = network(1)
        self.actor_mean = network(action_dim, 0.01 * np.sqrt(2))
        self.actor_logstd = nn.Parameter(torch.full((1, action_dim), -0.5))

    def get_value(self, observation):
        return self.critic(observation)

    def get_action(self, observation, deterministic=False):
        mean = self.actor_mean(observation)
        if deterministic:
            return mean
        return Normal(mean, self.actor_logstd.exp().expand_as(mean)).sample()

    def get_action_and_value(self, observation, action=None):
        mean = self.actor_mean(observation)
        distribution = Normal(mean, self.actor_logstd.exp().expand_as(mean))
        if action is None:
            action = distribution.sample()
        return (
            action,
            distribution.log_prob(action).sum(1),
            distribution.entropy().sum(1),
            self.critic(observation),
        )


def _atomic_torch_save(payload, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _rng_state():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    # ``torch.load(map_location="cuda")`` also maps this byte tensor, while
    # the default CPU generator explicitly requires a CPU ByteTensor.
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([rng_state.cpu() for rng_state in state["cuda"]])


def _checkpoint_payload(
    agent, optimizer, iteration, global_step, best_score, best_success, best_metrics, task,
):
    return {
        "schema_version": 1,
        "task": task,
        "agent": agent.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iteration": iteration,
        "global_step": global_step,
        "best_score": best_score,
        "best_success": best_success,
        "best_metrics": best_metrics,
        "rng": _rng_state(),
    }


def _select_task(config, task_index):
    tasks = [
        {**experiment, "seed": seed}
        for experiment in config["experiments"]
        for seed in config["seeds"]
    ]
    if not 0 <= task_index < len(tasks):
        raise ValueError(f"task-index must be in [0, {len(tasks) - 1}]")
    return tasks[task_index], len(tasks)


def _metric_success(metrics):
    for key in ("success_once", "success_at_end", "success"):
        if key in metrics:
            return float(metrics[key])
    return float("nan")


def _metric_failure(metrics):
    for key in ("fail_once", "fail_at_end", "fail"):
        if key in metrics:
            return float(metrics[key])
    return 0.0


def _environment_kwargs(task, evaluation=False):
    kwargs = {
        "obs_mode": task.get("obs_mode", "state"), "render_mode": None,
        "sim_backend": "physx_cuda", "control_mode": task["control_mode"],
        "reward_mode": task.get("reward_mode", "normalized_dense"),
    }
    kwargs.update(task.get("env_kwargs", {}))
    if evaluation:
        kwargs.update(task.get("eval_env_kwargs", {}))
    if "collision_stack_size" in task:
        kwargs["sim_config"] = {
            "gpu_memory_config": {
                "collision_stack_size": int(task["collision_stack_size"]),
            }
        }
    return kwargs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/manipulation_ppo_v1.json")
    parser.add_argument("--output", default="results/manipulation_ppo")
    parser.add_argument("--task-index", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", "0")))
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    task, task_count = _select_task(config, args.task_index)
    if args.preflight:
        print(json.dumps({"task_count": task_count, "task_index": args.task_index, **task}, indent=2))
        return
    if not torch.cuda.is_available():
        raise RuntimeError("manipulation PPO requires a CUDA GPU")
    if task.get("registration_module"):
        importlib.import_module(task["registration_module"])

    seed = int(task["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda")
    experiment_name = task.get("method", task["env_id"])
    run_dir = Path(args.output) / config["name"] / experiment_name / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "task.json").write_text(json.dumps(task, indent=2) + "\n", encoding="utf-8")

    env_kwargs = _environment_kwargs(task)
    eval_env_kwargs = _environment_kwargs(task, evaluation=True)
    envs = gym.make(task["env_id"], num_envs=task["num_envs"], **env_kwargs)
    eval_envs = gym.make(
        task["env_id"], num_envs=config["num_eval_envs"], reconfiguration_freq=1,
        **eval_env_kwargs,
    )
    if isinstance(envs.action_space, gym.spaces.Dict):
        envs = FlattenActionSpaceWrapper(envs)
        eval_envs = FlattenActionSpaceWrapper(eval_envs)
    envs = ManiSkillVectorEnv(envs, task["num_envs"], record_metrics=True)
    eval_envs = ManiSkillVectorEnv(
        eval_envs, config["num_eval_envs"], ignore_terminations=True, record_metrics=True,
    )
    observation_dim = int(np.prod(envs.single_observation_space.shape))
    action_dim = int(np.prod(envs.single_action_space.shape))
    agent = Agent(observation_dim, action_dim).to(device)
    optimizer = torch.optim.Adam(agent.parameters(), lr=config["learning_rate"], eps=1e-5)
    action_low = torch.as_tensor(envs.single_action_space.low, device=device)
    action_high = torch.as_tensor(envs.single_action_space.high, device=device)

    latest_path = run_dir / "latest.pt"
    best_path = run_dir / "best.pt"
    start_iteration, global_step = 1, 0
    best_score, best_success, best_metrics = float("-inf"), float("-inf"), {}
    if latest_path.exists():
        checkpoint = torch.load(latest_path, map_location=device, weights_only=False)
        if checkpoint["task"] != task:
            raise ValueError("checkpoint task does not match immutable task configuration")
        agent.load_state_dict(checkpoint["agent"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_iteration = int(checkpoint["iteration"]) + 1
        global_step = int(checkpoint["global_step"])
        best_success = float(checkpoint["best_success"])
        best_score = float(checkpoint.get("best_score", best_success))
        best_metrics = dict(checkpoint.get("best_metrics", {}))
        _restore_rng(checkpoint["rng"])
    elif task.get("init_checkpoint"):
        initialization_path = Path(str(task["init_checkpoint"]).format(seed=seed))
        if not initialization_path.exists():
            raise FileNotFoundError(
                f"initialization checkpoint unavailable: {initialization_path}"
            )
        initialization = torch.load(
            initialization_path, map_location=device, weights_only=False,
        )
        agent.load_state_dict(initialization["agent"], strict=True)
        (run_dir / "initialization.json").write_text(json.dumps({
            "checkpoint": str(initialization_path),
            "sha256": hashlib.sha256(initialization_path.read_bytes()).hexdigest(),
            "source_task": initialization.get("task"),
            "source_iteration": initialization.get("iteration"),
            "source_global_step": initialization.get("global_step"),
            "optimizer_reinitialized": True,
        }, indent=2) + "\n", encoding="utf-8")

    anchor_agent = None
    anchor_coefficient = float(config.get("anchor_actor_coefficient", 0.0))
    if anchor_coefficient > 0:
        if not task.get("init_checkpoint"):
            raise ValueError("anchor_actor_coefficient requires init_checkpoint")
        anchor_path = Path(str(task["init_checkpoint"]).format(seed=seed))
        anchor_checkpoint = torch.load(anchor_path, map_location=device, weights_only=False)
        anchor_agent = Agent(observation_dim, action_dim).to(device)
        anchor_agent.load_state_dict(anchor_checkpoint["agent"], strict=True)
        anchor_agent.eval()
        for parameter in anchor_agent.parameters():
            parameter.requires_grad_(False)

    num_envs, num_steps = int(task["num_envs"]), int(task["num_steps"])
    batch_size = num_envs * num_steps
    num_iterations = int(task["total_timesteps"]) // batch_size
    minibatch_size = batch_size // int(config["num_minibatches"])
    obs_shape, action_shape = envs.single_observation_space.shape, envs.single_action_space.shape
    obs = torch.zeros((num_steps, num_envs) + obs_shape, device=device)
    actions = torch.zeros((num_steps, num_envs) + action_shape, device=device)
    logprobs = torch.zeros((num_steps, num_envs), device=device)
    rewards = torch.zeros((num_steps, num_envs), device=device)
    dones = torch.zeros((num_steps, num_envs), device=device)
    values = torch.zeros((num_steps, num_envs), device=device)
    next_obs, _ = envs.reset(seed=seed)
    next_done = torch.zeros(num_envs, device=device)
    stop_requested = False

    def request_stop(_signum, _frame):
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGUSR1, request_stop)
    history_path = run_dir / "metrics.jsonl"
    started = time.time()
    for iteration in range(start_iteration, num_iterations + 1):
        if config.get("anneal_lr", False):
            fraction_remaining = 1.0 - (iteration - 1.0) / num_iterations
            optimizer.param_groups[0]["lr"] = fraction_remaining * config["learning_rate"]
        eval_success = float("nan")
        if iteration == start_iteration or iteration % int(config["eval_freq"]) == 0:
            eval_obs, _ = eval_envs.reset(seed=seed + iteration)
            eval_metrics = defaultdict(list)
            with torch.no_grad():
                for _ in range(int(task["num_eval_steps"])):
                    eval_obs, _, _, _, info = eval_envs.step(agent.get_action(eval_obs, True))
                    if "final_info" in info:
                        mask = info["_final_info"]
                        for key, value in info["final_info"]["episode"].items():
                            eval_metrics[key].append(value[mask].float())
            means = {
                key: float(torch.cat(value).mean().item())
                for key, value in eval_metrics.items() if value and torch.cat(value).numel()
            }
            eval_success = _metric_success(means)
            eval_failure = _metric_failure(means)
            eval_return = float(means.get("return", float("-inf")))
            # By default success is primary and return only breaks ties. Safe
            # experiments can predeclare a failure penalty, making checkpoint
            # selection optimize task completion and hard-constraint safety.
            eval_score = (
                eval_success
                - float(config.get("selection_failure_penalty", 0.0)) * eval_failure
                + 1e-6 * eval_return
            )
            with history_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({
                    "iteration": iteration, "global_step": global_step,
                    "eval": means, "elapsed_seconds": time.time() - started,
                }) + "\n")
            # Save the weights that were actually evaluated, before the next
            # rollout/update changes them. ``iteration - 1`` is the number of
            # completed optimizer iterations represented by this checkpoint.
            if np.isfinite(eval_score) and eval_score > best_score:
                best_score, best_success, best_metrics = eval_score, eval_success, means
                payload = _checkpoint_payload(
                    agent, optimizer, iteration - 1, global_step,
                    best_score, best_success, best_metrics, task,
                )
                _atomic_torch_save(payload, best_path)

        final_values = torch.zeros((num_steps, num_envs), device=device)
        agent.eval()
        for step in range(num_steps):
            global_step += num_envs
            obs[step], dones[step] = next_obs, next_done
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step], logprobs[step] = action, logprob
            next_obs, reward, terminated, truncated, info = envs.step(
                torch.clamp(action.detach(), action_low, action_high)
            )
            next_done = (terminated | truncated).float()
            rewards[step] = reward.view(-1)
            if "final_info" in info:
                mask = info["_final_info"]
                with torch.no_grad():
                    final_values[step, torch.arange(num_envs, device=device)[mask]] = (
                        agent.get_value(info["final_observation"][mask]).view(-1)
                    )

        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards)
            last_gae = 0
            for t in reversed(range(num_steps)):
                if t == num_steps - 1:
                    next_not_done, next_values = 1.0 - next_done, next_value
                else:
                    next_not_done, next_values = 1.0 - dones[t + 1], values[t + 1]
                real_next = next_not_done * next_values + final_values[t]
                delta = rewards[t] + config["gamma"] * real_next - values[t]
                advantages[t] = last_gae = (
                    delta + config["gamma"] * config["gae_lambda"] * next_not_done * last_gae
                )
            returns = advantages + values

        b_obs = obs.reshape((-1,) + obs_shape)
        b_actions = actions.reshape((-1,) + action_shape)
        b_logprobs = logprobs.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        indices = np.arange(batch_size)
        agent.train()
        for _ in range(int(config["update_epochs"])):
            np.random.shuffle(indices)
            for start in range(0, batch_size, minibatch_size):
                mb = indices[start:start + minibatch_size]
                _, new_logprob, entropy, new_value = agent.get_action_and_value(b_obs[mb], b_actions[mb])
                logratio = new_logprob - b_logprobs[mb]
                ratio = logratio.exp()
                with torch.no_grad():
                    approximate_kl = ((ratio - 1.0) - logratio).mean()
                target_kl = config.get("target_kl")
                if target_kl is not None and approximate_kl > float(target_kl):
                    break
                adv = b_advantages[mb]
                adv = (adv - adv.mean()) / (adv.std() + 1e-8)
                pg_loss = torch.maximum(
                    -adv * ratio,
                    -adv * torch.clamp(ratio, 0.8, 1.2),
                ).mean()
                value_loss = 0.5 * ((new_value.view(-1) - b_returns[mb]) ** 2).mean()
                loss = (
                    pg_loss
                    - float(config.get("entropy_coefficient", 0.0)) * entropy.mean()
                    + float(config.get("value_coefficient", 0.5)) * value_loss
                )
                if anchor_agent is not None:
                    loss = loss + anchor_coefficient * torch.nn.functional.mse_loss(
                        agent.actor_mean(b_obs[mb]),
                        anchor_agent.actor_mean(b_obs[mb]),
                    )
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
                optimizer.step()
            if target_kl is not None and approximate_kl > float(target_kl):
                break

        should_save = (
            iteration % int(config["checkpoint_freq"]) == 0
            or iteration == num_iterations or stop_requested
        )
        if should_save:
            payload = _checkpoint_payload(
                agent, optimizer, iteration, global_step,
                best_score, best_success, best_metrics, task,
            )
            _atomic_torch_save(payload, latest_path)
        if stop_requested:
            break

    envs.close()
    eval_envs.close()
    print(json.dumps({
        "env_id": task["env_id"], "seed": seed, "global_step": global_step,
        "scheduled_global_steps": num_iterations * batch_size,
        "requested_timesteps": int(task["total_timesteps"]),
        "best_success": best_success, "best_metrics": best_metrics,
        "latest_checkpoint": str(latest_path),
        # PPO updates are whole vector batches, so a requested budget that is
        # not batch-divisible intentionally ends at floor(T / batch) batches.
        "completed": global_step >= num_iterations * batch_size,
    }, indent=2))


if __name__ == "__main__":
    main()

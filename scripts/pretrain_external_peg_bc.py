#!/usr/bin/env python3
"""Pretrain the shared PegInsertion PPO actor from official RL demonstrations."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from train_manipulation_ppo import Agent


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def episode_bucket(episode_id: int) -> int:
    value = hashlib.sha256(str(episode_id).encode()).hexdigest()[:8]
    return int(value, 16) % 10


def load_demonstrations(path: Path, count: int):
    if path.suffix == ".npz":
        archive = np.load(path)
        episode_id = archive["episode_id"].astype(np.int64)
        keep_episodes = np.unique(episode_id)[:count]
        keep = np.isin(episode_id, keep_episodes)
        validation = np.array([
            episode_bucket(value) == 0 for value in episode_id
        ]) & keep
        train = keep & ~validation
        return (
            archive["observation"][train].astype(np.float32),
            archive["action"][train].astype(np.float32),
            archive["observation"][validation].astype(np.float32),
            archive["action"][validation].astype(np.float32),
            int(len(np.unique(episode_id[train]))),
            int(len(np.unique(episode_id[validation]))),
        )
    train_observations, train_actions = [], []
    validation_observations, validation_actions = [], []
    train_episodes, validation_episodes = 0, 0
    with h5py.File(path, "r") as archive:
        names = sorted(archive, key=lambda name: int(name.split("_")[-1]))[:count]
        for name in names:
            episode_id = int(name.split("_")[-1])
            group = archive[name]
            observation = np.asarray(group["obs"], dtype=np.float32)
            action = np.asarray(group["actions"], dtype=np.float32)
            if observation.ndim != 2 or len(observation) != len(action) + 1:
                raise ValueError(f"{name} has an invalid flat-state trajectory contract")
            if action.ndim != 2:
                raise ValueError(f"{name} actions are not a matrix")
            if episode_bucket(episode_id) == 0:
                validation_observations.append(observation[:-1])
                validation_actions.append(action)
                validation_episodes += 1
            else:
                train_observations.append(observation[:-1])
                train_actions.append(action)
                train_episodes += 1
    if not train_observations or not validation_observations:
        raise RuntimeError("episode-disjoint train/validation split is empty")
    return (
        np.concatenate(train_observations), np.concatenate(train_actions),
        np.concatenate(validation_observations), np.concatenate(validation_actions),
        train_episodes, validation_episodes,
    )


@torch.inference_mode()
def validation_mse(agent: Agent, observation: torch.Tensor, action: torch.Tensor) -> float:
    agent.eval()
    squared_error = []
    for start in range(0, len(observation), 8192):
        prediction = agent.actor_mean(observation[start:start + 8192])
        squared_error.append(F.mse_loss(
            prediction, action[start:start + 8192], reduction="sum",
        ))
    return float(torch.stack(squared_error).sum() / action.numel())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--count", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("behavioral pretraining requires CUDA")
    device = torch.device("cuda")
    (
        train_observation, train_action, validation_observation, validation_action,
        train_episodes, validation_episodes,
    ) = load_demonstrations(args.data, args.count)
    train_dataset = TensorDataset(
        torch.from_numpy(train_observation), torch.from_numpy(train_action),
    )
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        generator=generator, pin_memory=True,
    )
    validation_observation_tensor = torch.from_numpy(validation_observation).to(device)
    validation_action_tensor = torch.from_numpy(validation_action).to(device)
    agent = Agent(
        train_observation.shape[1], train_action.shape[1], actor_logstd_initial=0.0,
        fast_action_sampling=True,
    ).to(device)
    optimizer = torch.optim.Adam(agent.actor_mean.parameters(), lr=args.learning_rate)
    best_mse = float("inf")
    best_epoch = -1
    best_state = None
    history = []
    for epoch in range(args.epochs):
        agent.train()
        total_squared_error = 0.0
        elements = 0
        for observation, action in loader:
            observation = observation.to(device, non_blocking=True)
            action = action.to(device, non_blocking=True)
            prediction = agent.actor_mean(observation)
            loss = F.mse_loss(prediction, action)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.actor_mean.parameters(), 5.0)
            optimizer.step()
            total_squared_error += float(loss.detach()) * action.numel()
            elements += action.numel()
        observed_validation_mse = validation_mse(
            agent, validation_observation_tensor, validation_action_tensor,
        )
        history.append({
            "epoch": epoch,
            "train_mse": total_squared_error / elements,
            "validation_mse": observed_validation_mse,
        })
        if observed_validation_mse < best_mse:
            best_mse = observed_validation_mse
            best_epoch = epoch
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in agent.state_dict().items()
            }
    payload = {
        "schema_version": 1,
        "method": "official_rl_demo_behavioral_initialization",
        "seed": args.seed,
        "agent": best_state,
        "observation_dim": int(train_observation.shape[1]),
        "action_dim": int(train_action.shape[1]),
        "data": str(args.data),
        "data_sha256": file_sha256(args.data),
        "demonstration_count": args.count,
        "train_episodes": train_episodes,
        "validation_episodes": validation_episodes,
        "split": "sha256(episode_id) modulo 10; bucket 0 validation",
        "best_epoch": best_epoch,
        "best_validation_mse": best_mse,
        "history": history,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(args.output)
    summary_path = args.output.with_suffix(".json")
    summary_path.write_text(json.dumps({
        key: value for key, value in payload.items()
        if key not in {"agent", "history"}
    }, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "seed": args.seed,
        "best_epoch": best_epoch,
        "best_validation_mse": best_mse,
        "train_episodes": train_episodes,
        "validation_episodes": validation_episodes,
        "output": str(args.output),
    }, indent=2))


if __name__ == "__main__":
    main()

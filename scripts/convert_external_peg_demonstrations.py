#!/usr/bin/env python3
"""Convert official PegInsertion simulator states into flat state-action rows."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import gymnasium as gym
import h5py
import numpy as np
import torch

import mani_skill.envs  # noqa: F401


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_actor_name(name: str) -> str:
    return name[:-2] if name.endswith("_0") else name


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=1000)
    parser.add_argument("--num-envs", type=int, default=1024)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("demonstration state conversion requires CUDA")
    actions, episode_ids, steps = [], [], []
    state_rows: dict[str, dict[str, list[np.ndarray]]] = {
        "actors": {}, "articulations": {},
    }
    with h5py.File(args.source, "r") as archive:
        names = sorted(archive, key=lambda name: int(name.split("_")[-1]))[:args.count]
        for name in names:
            episode_id = int(name.split("_")[-1])
            group = archive[name]
            action = np.asarray(group["actions"], dtype=np.float32)
            actions.append(action)
            episode_ids.append(np.full(len(action), episode_id, dtype=np.int64))
            steps.append(np.arange(len(action), dtype=np.int64))
            for family in state_rows:
                for raw_name, dataset in group["env_states"][family].items():
                    state_name = (
                        canonical_actor_name(raw_name)
                        if family == "actors" else raw_name
                    )
                    state_rows[family].setdefault(state_name, []).append(
                        np.asarray(dataset[:len(action)], dtype=np.float32)
                    )
    action_array = np.concatenate(actions)
    episode_id_array = np.concatenate(episode_ids)
    step_array = np.concatenate(steps)
    states = {
        family: {name: np.concatenate(parts) for name, parts in entries.items()}
        for family, entries in state_rows.items()
    }
    env = gym.make(
        "PegInsertionSide-v1", num_envs=args.num_envs, reconfiguration_freq=1,
        obs_mode="state", render_mode=None, sim_backend="physx_cuda",
        control_mode="pd_joint_delta_pos", reward_mode="normalized_dense",
    )
    base = env.unwrapped
    env.reset(seed=0)
    observations = []
    total = len(action_array)
    try:
        for start in range(0, total, args.num_envs):
            stop = min(start + args.num_envs, total)
            count = stop - start
            batch = {}
            for family, entries in states.items():
                batch[family] = {}
                for name, value in entries.items():
                    selected = value[start:stop]
                    if count < args.num_envs:
                        selected = np.concatenate((
                            selected,
                            np.repeat(selected[-1:], args.num_envs - count, axis=0),
                        ))
                    batch[family][name] = torch.from_numpy(selected).cuda()
            base.set_state_dict(batch)
            observation = base.get_obs()
            if not isinstance(observation, torch.Tensor) or observation.ndim != 2:
                raise ValueError("state observation is not a flat tensor matrix")
            observations.append(observation[:count].cpu().numpy().astype(np.float32))
    finally:
        env.close()
    observation_array = np.concatenate(observations)
    if len(observation_array) != len(action_array):
        raise RuntimeError("converted observation/action row counts differ")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output, observation=observation_array, action=action_array,
        episode_id=episode_id_array, step=step_array,
    )
    metadata = {
        "schema_version": 1,
        "source": str(args.source),
        "source_sha256": file_sha256(args.source),
        "environment": "PegInsertionSide-v1",
        "obs_mode": "state",
        "control_mode": "pd_joint_delta_pos",
        "demonstration_count": args.count,
        "rows": total,
        "observation_dim": int(observation_array.shape[1]),
        "action_dim": int(action_array.shape[1]),
        "conversion": "exact downloaded simulator states queried through pinned environment",
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

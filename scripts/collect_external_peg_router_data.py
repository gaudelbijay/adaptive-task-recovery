#!/usr/bin/env python3
"""Collect group-disjoint causal prefixes for the external Peg router."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

import mani_skill.envs  # noqa: F401
import atr.envs.peg_insertion_recovery  # noqa: F401
from atr.policies.peg_router_features import (
    GEOMETRY_NAMES, LATERAL_Y_INDICES, relative_geometry,
)
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

from train_manipulation_ppo import Agent


KINDS = (
    "nominal", "positive_lateral_peg_ejection", "permanent_hole_block",
    "temporary_hole_block", "negative_lateral_peg_ejection",
)
SNAPSHOTS = (1, 2, 3, 4, 8, 12, 16, 24, 32, 40, 48, 64, 80, 96)


def make_env(
    num_envs: int,
    kind: str,
    include_blocker_state_observation: bool,
    ejection_force: float,
    ejection_steps: int,
    ejection_target_displacement: float,
    ejection_position_gain: float,
    ejection_velocity_gain: float,
):
    kwargs = dict(
        num_envs=num_envs, obs_mode="state", render_mode=None,
        sim_backend="physx_cuda", control_mode="pd_joint_delta_pos",
        reward_mode="normalized_dense", reconfiguration_freq=1,
        intervention_probability=0.0 if kind == "nominal" else 1.0,
        intervention_types=(
            ("positive_lateral_peg_ejection",) if kind == "nominal" else (kind,)
        ),
        onset_step_range=(18, 42),
        include_blocker_state_observation=include_blocker_state_observation,
        ejection_force=ejection_force,
        ejection_steps=ejection_steps,
        ejection_target_displacement=ejection_target_displacement,
        ejection_position_gain=ejection_position_gain,
        ejection_velocity_gain=ejection_velocity_gain,
    )
    env = gym.make("PegInsertionRecovery-v1", **kwargs)
    if isinstance(env.action_space, gym.spaces.Dict):
        env = FlattenActionSpaceWrapper(env)
    return ManiSkillVectorEnv(env, num_envs, ignore_terminations=True, record_metrics=False)


def labels(
    kind_index: int, info: dict, length: int, geometry_history: torch.Tensor,
):
    onset = info["critic_intervention_onset_step"].long()
    post_event = length >= onset + 2
    count = len(onset)
    option = torch.full(
        (count,), {0: 0, 1: 1, 2: 3, 3: 4, 4: 2}[kind_index],
        dtype=torch.long, device=onset.device,
    )
    event = torch.zeros(count, dtype=torch.long, device=onset.device)
    direction = torch.full((count,), -100, dtype=torch.long, device=onset.device)
    block = torch.full_like(direction, -100)
    ready = torch.zeros(count, dtype=torch.bool, device=onset.device)
    if kind_index == 0:
        option.zero_(); ready.fill_(True)
    elif kind_index in (1, 4):
        event[post_event] = 1
        direction[post_event] = 0 if kind_index == 1 else 1
        # Readiness must be justified by the causal prefix, not merely by an
        # oracle clock. Compare the current hole-frame lateral position with
        # the prefix frame at onset and require motion in the labeled physical
        # direction. Onset is a training-only alignment target and never an
        # input to the deployed router.
        reference_index = onset.clamp(min=1, max=length) - 1
        row = torch.arange(count, device=onset.device)
        reference_lateral = geometry_history[row, reference_index, 1]
        sign = 1.0 if kind_index == 1 else -1.0
        directed_history = sign * (
            geometry_history[:, :, 1] - reference_lateral[:, None]
        )
        frame_index = torch.arange(length, device=onset.device)[None, :]
        directed_history = directed_history.masked_fill(
            frame_index < reference_index[:, None], -torch.inf,
        )
        ready = post_event & (directed_history.max(dim=1).values > 0.01)
    elif kind_index == 2:
        event[post_event] = 2
        engaged = info["blocker_engaged"].bool()
        block[engaged] = 1
        ready = engaged
    elif kind_index == 3:
        event[post_event] = 2
        cleared = info["temporary_cleared"].bool()
        block[cleared] = 2
        ready = cleared
    return option, event, direction, block, ready, onset


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata-output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--batches-per-kind", type=int, default=12)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--horizon", type=int, default=96)
    parser.add_argument("--seed-base", type=int, default=421_100_000)
    parser.add_argument("--ejection-force", type=float, default=12.0)
    parser.add_argument("--ejection-steps", type=int, default=30)
    parser.add_argument("--ejection-target-displacement", type=float, default=0.06)
    parser.add_argument("--ejection-position-gain", type=float, default=160.0)
    parser.add_argument("--ejection-velocity-gain", type=float, default=10.0)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("external Peg prefix collection requires CUDA")
    checkpoint = torch.load(args.checkpoint, map_location="cuda", weights_only=False)
    recovery_kwargs = checkpoint["task"].get(
        "competence_env_kwargs", checkpoint["task"].get("env_kwargs", {})
    )
    include_blocker_state_observation = bool(
        recovery_kwargs.get(
            "include_blocker_state_observation", True,
        )
    )
    rows = {key: [] for key in (
        "sequence", "length", "option", "event", "direction", "block_status",
        "temporary_cleared", "option_ready", "onset", "return_delay",
        "group_id", "condition", "physical_heldout", "counterfactual_reflection",
    )}
    for kind_index, kind in enumerate(KINDS):
        for batch in range(args.batches_per_kind):
            env = make_env(
                args.num_envs,
                kind,
                include_blocker_state_observation,
                args.ejection_force,
                args.ejection_steps,
                args.ejection_target_displacement,
                args.ejection_position_gain,
                args.ejection_velocity_gain,
            )
            try:
                observation, info = env.reset(
                    seed=args.seed_base + kind_index * 10_000 + batch,
                )
                agent = Agent(
                    int(np.prod(env.single_observation_space.shape)),
                    int(np.prod(env.single_action_space.shape)),
                ).cuda().eval()
                agent.load_state_dict(checkpoint["agent"], strict=True)
                action_low = torch.as_tensor(env.single_action_space.low, device="cuda")
                action_high = torch.as_tensor(env.single_action_space.high, device="cuda")
                history = []
                for step in range(1, args.horizon + 1):
                    geometry = relative_geometry(info["router_task_geometry"])
                    normalized_time = torch.full(
                        (args.num_envs, 1), step / args.horizon,
                        dtype=geometry.dtype, device=geometry.device,
                    )
                    feature = torch.cat((geometry, normalized_time), dim=1)
                    history.append(feature)
                    if step in SNAPSHOTS:
                        padded = torch.zeros(
                            args.num_envs, args.horizon, feature.shape[1],
                            dtype=feature.dtype, device=feature.device,
                        )
                        geometry_history = torch.stack(history, dim=1)
                        padded[:, :step] = geometry_history
                        option, event, direction, block, ready, onset = labels(
                            kind_index, info, step, geometry_history,
                        )
                        rows["sequence"].append(padded.cpu().numpy().astype(np.float32))
                        rows["length"].append(np.full(args.num_envs, step, dtype=np.int64))
                        for name, tensor in (
                            ("option", option), ("event", event),
                            ("direction", direction), ("block_status", block),
                            ("option_ready", ready), ("onset", onset),
                        ):
                            rows[name].append(tensor.cpu().numpy())
                        rows["temporary_cleared"].append(
                            info["temporary_cleared"].bool().cpu().numpy()
                        )
                        rows["return_delay"].append(np.full(
                            args.num_envs, 48, dtype=np.int64,
                        ))
                        rows["group_id"].append(np.full(
                            args.num_envs, kind_index * 1_000_000 + batch,
                            dtype=np.int64,
                        ))
                        rows["condition"].append(np.full(
                            args.num_envs, kind_index, dtype=np.int64,
                        ))
                        rows["physical_heldout"].append(np.full(
                            args.num_envs, kind_index == 4, dtype=np.bool_,
                        ))
                        rows["counterfactual_reflection"].append(np.zeros(
                            args.num_envs, dtype=np.bool_,
                        ))
                    action = torch.clamp(
                        agent.get_action(observation, deterministic=True),
                        action_low, action_high,
                    )
                    observation, _, _, _, info = env.step(action)
            finally:
                env.close()

    packed = {name: np.concatenate(parts) for name, parts in rows.items()}
    positive = (packed["condition"] == 1) & ~packed["physical_heldout"]
    reflected = {name: value[positive].copy() for name, value in packed.items()}
    # Feature layout is four relative xyz vectors plus normalized time. A
    # reflection across the task's lateral y-axis negates indices 1,4,7,10.
    reflected["sequence"][:, :, list(LATERAL_Y_INDICES)] *= -1
    reflected["option"].fill(2)
    post_event = reflected["direction"] >= 0
    reflected["direction"][post_event] = 1
    reflected["counterfactual_reflection"].fill(True)
    reflected["physical_heldout"].fill(False)
    packed = {
        name: np.concatenate((value, reflected[name]))
        for name, value in packed.items()
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **packed)
    metadata = {
        "schema_version": 1,
        "environment": "PegInsertionRecovery-v1",
        "conditions": list(KINDS),
        "rows": int(len(packed["length"])),
        "simulator_batch_groups": int(len(np.unique(packed["group_id"]))),
        "feature_names": [*GEOMETRY_NAMES, "normalized_time"],
        # Preserve the full current hole-relative geometry for the static
        # baseline.  A zero here means no additional current-centering is
        # applied by the trainer; the features are already expressed in the
        # randomized hole frame.
        "current_centered_geometry_dim": 0,
        "forbidden_feature_keys": [
            "critic_intervention_mechanism", "critic_intervention_onset_step",
            "critic_physical_unavailable", "success", "future_observation",
        ],
        "training_only_targets": [
            "event_family", "physical_direction", "blocker_engaged",
            "temporary_cleared", "option_ready",
        ],
        "sweep_readiness_rule": (
            "more than 1 cm maximum prefix-observed displacement in the "
            "labeled hole-frame direction relative to the onset-aligned "
            "past frame"
        ),
        "heldout_option": 2,
        "heldout_option_cross_entropy": False,
        "real_negative_ejection_split": "physical_heldout test-only",
        "counterfactual_reflection": {
            "source": "positive_lateral_peg_ejection factual prefixes",
            "reflected_feature_indices": list(LATERAL_Y_INDICES),
            "frame": "randomized hole-local frame",
            "shared_group_with_factual": True,
            "option_cross_entropy": False,
            "rows": int(packed["counterfactual_reflection"].sum()),
        },
        "prefix_timestamp": "pre_action_observation_matching_deployment",
        "split_unit": "entire vectorized simulator reset batch",
        "snapshots": [step for step in SNAPSHOTS if step <= args.horizon],
        "absolute_pose_features": False,
        "hand_engineered_temporal_features": False,
        "collection_policy": "deterministic official-style nominal state PPO",
        "seed_base": args.seed_base,
        "behavior_checkpoint": str(args.checkpoint),
        "behavior_checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "include_blocker_state_observation": include_blocker_state_observation,
        "ejection_force": args.ejection_force,
        "ejection_steps": args.ejection_steps,
        "ejection_target_displacement": args.ejection_target_displacement,
        "ejection_position_gain": args.ejection_position_gain,
        "ejection_velocity_gain": args.ejection_velocity_gain,
    }
    args.metadata_output.parent.mkdir(parents=True, exist_ok=True)
    args.metadata_output.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: v for k, v in metadata.items() if k != "feature_names"}, indent=2))


if __name__ == "__main__":
    main()

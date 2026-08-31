#!/usr/bin/env python3
"""Pad a V3/V1 state policy's input layers for V4's appended blocker poses."""

import argparse
import copy
import hashlib
import os
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--target-observation-dim", type=int, default=84)
    args = parser.parse_args(); source = Path(args.source); output = Path(args.output)
    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    adapted = copy.deepcopy(checkpoint)
    for key in ("actor_mean.0.weight", "critic.0.weight"):
        old = checkpoint["agent"][key]
        if old.shape[1] >= args.target_observation_dim:
            raise ValueError(f"{key} is not narrower than the requested target")
        padded = old.new_zeros((old.shape[0], args.target_observation_dim))
        padded[:, :old.shape[1]] = old
        adapted["agent"][key] = padded
    adapted.pop("optimizer", None)
    adapted["input_adapter"] = {
        "kind": "append_zero_initialized_v4_goal_blocker_poses",
        "source_observation_dim": int(checkpoint["agent"]["actor_mean.0.weight"].shape[1]),
        "target_observation_dim": args.target_observation_dim,
        "source_checkpoint": str(source),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp.{os.getpid()}")
    torch.save(adapted, temporary); os.replace(temporary, output)
    print(adapted["input_adapter"])


if __name__ == "__main__": main()

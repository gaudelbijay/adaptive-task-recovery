#!/usr/bin/env python3
"""Calibrate early sweep dispatch on group-disjoint router validation rows."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from atr.policies.causal_option_router import (
    CausalOptionRouter,
    StaticOptionRouter,
    causal_safe_targets,
    current_centered_sequence,
)
from train_v4_causal_option_router import group_split


def load_model(checkpoint: dict, device: torch.device):
    if checkpoint["model"] == "causal_gru":
        model = CausalOptionRouter(
            checkpoint["input_dim"], checkpoint["hidden_dim"], 2,
        )
    elif checkpoint["model"] == "static_mlp":
        model = StaticOptionRouter(checkpoint["input_dim"], checkpoint["hidden_dim"])
    else:
        raise ValueError("factorized calibration requires a structured router")
    model.load_state_dict(checkpoint["state_dict"])
    return model.to(device).eval()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--maximum-error", type=float, default=0.01)
    args = parser.parse_args()
    source_hash = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint["data_sha256"] != hashlib.sha256(args.data.read_bytes()).hexdigest():
        raise ValueError("router data hash mismatch")
    if checkpoint["feature_metadata_sha256"] != hashlib.sha256(
        args.metadata.read_bytes()
    ).hexdigest():
        raise ValueError("router metadata hash mismatch")
    raw = np.load(args.data)
    tensors = {key: torch.from_numpy(raw[key]) for key in raw.files}
    safe_option, _ = causal_safe_targets(tensors)
    _, validation, _ = group_split(raw["group_id"])
    indices = torch.from_numpy(np.flatnonzero(validation))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(checkpoint, device)
    event_probabilities = []
    direction_probabilities = []
    directions = []
    targets = []
    with torch.inference_mode():
        for start in range(0, len(indices), 512):
            index = indices[start:start + 512]
            sequence = tensors["sequence"][index].to(device)
            length = tensors["length"][index].to(device)
            sequence = current_centered_sequence(
                sequence, length,
                int(checkpoint.get("current_centered_geometry_dim", 0)),
            )
            output = model(sequence, length)
            event_probabilities.append(output.event_logits.softmax(-1)[:, 1].cpu())
            confidence, direction = output.direction_logits.softmax(-1).max(1)
            direction_probabilities.append(confidence.cpu())
            directions.append(direction.cpu())
            targets.append(safe_option[index])
    event_probability = torch.cat(event_probabilities)
    direction_probability = torch.cat(direction_probabilities)
    predicted_option = torch.cat(directions) + 1
    target = torch.cat(targets)
    best = None
    grid = torch.linspace(0.5, 0.999, 101)
    for event_threshold in grid:
        event_selected = event_probability >= event_threshold
        for direction_threshold in grid:
            selected = event_selected & (direction_probability >= direction_threshold)
            count = int(selected.sum())
            if count < 10:
                continue
            error = float((predicted_option[selected] != target[selected]).float().mean())
            if error <= args.maximum_error:
                candidate = (
                    count, -error, -float(event_threshold),
                    -float(direction_threshold),
                )
                if best is None or candidate > best[0]:
                    best = (candidate, float(event_threshold), float(direction_threshold), error)
    if best is None:
        raise RuntimeError("no factorized sweep calibration meets precision target")
    count = best[0][0]
    calibration = {
        "event_threshold": best[1],
        "direction_threshold": best[2],
        "validation_dispatches": count,
        "validation_rows": len(indices),
        "validation_coverage": count / len(indices),
        "validation_error": best[3],
        "maximum_error": args.maximum_error,
        "split": "group_disjoint_validation",
    }
    checkpoint["calibration"] = dict(checkpoint["calibration"])
    checkpoint["calibration"]["factorized_sweep_dispatch_99_precision"] = calibration
    checkpoint["calibration_source_checkpoint_sha256"] = source_hash
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.output)
    result = {
        "source_checkpoint_sha256": source_hash,
        "output_checkpoint": str(args.output),
        "output_checkpoint_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
        "calibration": calibration,
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

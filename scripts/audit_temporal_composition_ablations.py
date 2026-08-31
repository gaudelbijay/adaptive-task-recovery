#!/usr/bin/env python3
"""Run history and current-frame ablations on held-out reverse prefixes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from atr.policies.causal_option_router import (
    CausalOptionRouter, causal_safe_targets, current_centered_sequence,
)
from train_v4_causal_option_router import group_split


def reverse_valid_prefix(sequence: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    result = sequence.clone()
    for row, length in enumerate(lengths.tolist()):
        result[row, :length] = result[row, :length].flip(0)
    return result


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=512)
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("heldout_option") != 2:
        raise RuntimeError("checkpoint was not trained with reverse option held out")
    geometry_dim = int(checkpoint.get("current_centered_geometry_dim", 0))
    if geometry_dim != 57:
        raise RuntimeError("checkpoint does not use the complete 57-D geometry contract")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CausalOptionRouter(
        checkpoint["input_dim"], checkpoint["hidden_dim"], 2,
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    raw = np.load(args.data)
    tensors = {key: torch.from_numpy(raw[key]) for key in raw.files}
    tensors["option"], _ = causal_safe_targets(tensors)
    _, _, test = group_split(raw["group_id"])
    selected = np.flatnonzero(test & (tensors["option"].numpy() == 2))
    loader = DataLoader(TensorDataset(torch.from_numpy(selected)), batch_size=args.batch_size)
    correct = {"clean": 0, "reversed_history": 0, "geometry_history_removed": 0}
    total = 0
    final_geometry_max_abs = 0.0
    digest = hashlib.sha256()
    for (index,) in loader:
        sequence = tensors["sequence"][index].to(device)
        lengths = tensors["length"][index].to(device)
        centered = current_centered_sequence(sequence, lengths, geometry_dim)
        current = centered[
            torch.arange(len(centered), device=device), lengths - 1, :geometry_dim,
        ]
        final_geometry_max_abs = max(final_geometry_max_abs, float(current.abs().max()))
        digest.update(centered[:, :, :geometry_dim].cpu().numpy().tobytes())
        variants = {
            "clean": centered,
            "reversed_history": reverse_valid_prefix(centered, lengths),
            "geometry_history_removed": centered.clone(),
        }
        variants["geometry_history_removed"][:, :, :geometry_dim] = 0
        for name, value in variants.items():
            prediction = model(value, lengths).option
            correct[name] += int((prediction == 2).sum())
        total += len(index)
    payload = {
        "schema_version": 1,
        "heldout_option": "reverse",
        "rows": total,
        "accuracy": {name: count / total for name, count in correct.items()},
        "final_geometry_max_abs": final_geometry_max_abs,
        "matched_centered_geometry_sha256": digest.hexdigest(),
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "data_sha256": checkpoint["data_sha256"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

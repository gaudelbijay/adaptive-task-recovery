#!/usr/bin/env python3
"""Train factorized and matched-baseline routers on disjoint episodes."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from atr.policies.causal_option_router import (
    CausalOptionRouter, StaticOptionRouter, UnstructuredOptionGRU,
    causal_safe_targets,
)


def group_split(group_id: np.ndarray):
    # Stable hash split; every prefix from an episode remains together.
    bucket = np.array([
        int(hashlib.sha256(str(int(x)).encode()).hexdigest()[:8], 16) % 100
        for x in group_id
    ])
    return bucket < 70, (bucket >= 70) & (bucket < 85), bucket >= 85


def make_model(name: str, input_dim: int, hidden_dim: int):
    if name == "causal_gru": return CausalOptionRouter(input_dim, hidden_dim, 2)
    if name == "static_mlp": return StaticOptionRouter(input_dim, hidden_dim)
    if name == "unstructured_gru": return UnstructuredOptionGRU(input_dim, hidden_dim, 2)
    raise ValueError(name)


def option_logp(model, sequence, length):
    output = model(sequence, length)
    return output.option_log_probability if hasattr(output, "option_log_probability") else output


def loss_for(model, batch, structured: bool):
    sequence, length, option, event, direction, block = batch
    output = model(sequence, length)
    logp = output.option_log_probability if structured else output
    loss = F.nll_loss(logp, option)
    if structured:
        readiness = (option != 5).long()
        loss = loss + 0.35 * F.cross_entropy(output.readiness_logits, readiness)
        loss = loss + 0.35 * F.cross_entropy(output.event_logits, event)
        sweep = direction >= 0
        blocked = block >= 0
        if bool(sweep.any()):
            loss = loss + 0.2 * F.cross_entropy(output.direction_logits[sweep], direction[sweep])
        if bool(blocked.any()):
            loss = loss + 0.25 * F.cross_entropy(output.block_status_logits[blocked], block[blocked])
    return loss


@torch.inference_mode()
def evaluate(model, tensors, mask, device):
    indices = torch.from_numpy(np.flatnonzero(mask))
    loader = DataLoader(TensorDataset(indices), batch_size=512)
    probabilities, targets, conditions, lengths = [], [], [], []
    model.eval()
    for (index,) in loader:
        sequence = tensors["sequence"][index].to(device)
        length = tensors["length"][index].to(device)
        probabilities.append(option_logp(model, sequence, length).exp().cpu())
        targets.append(tensors["option"][index])
        conditions.append(tensors["condition"][index])
        lengths.append(length.cpu())
    probability = torch.cat(probabilities)
    target = torch.cat(targets)
    condition = torch.cat(conditions)
    length = torch.cat(lengths)
    prediction = probability.argmax(1)
    result = {
        "rows": int(len(target)),
        "accuracy": float((prediction == target).float().mean()),
        "nll": float(F.nll_loss(probability.clamp_min(1e-9).log(), target)),
        "by_condition": {}, "by_horizon": {}, "by_condition_horizon": {},
    }
    for value in condition.unique().tolist():
        selected = condition == value
        result["by_condition"][str(value)] = float((prediction[selected] == target[selected]).float().mean())
    for value in length.unique().tolist():
        selected = length == value
        result["by_horizon"][str(value)] = float((prediction[selected] == target[selected]).float().mean())
    for condition_value in condition.unique().tolist():
        for length_value in length.unique().tolist():
            selected = (condition == condition_value) & (length == length_value)
            if bool(selected.any()):
                result["by_condition_horizon"][f"{condition_value}:{length_value}"] = float(
                    (prediction[selected] == target[selected]).float().mean()
                )
    return result, probability, target


def calibrate(probability: torch.Tensor, target: torch.Tensor):
    confidence, prediction = probability.max(1)
    candidates = []
    for threshold in torch.linspace(0.5, 0.99, 50):
        selected = confidence >= threshold
        if not bool(selected.any()): continue
        error = float((prediction[selected] != target[selected]).float().mean())
        coverage = float(selected.float().mean())
        candidates.append((error <= 0.01, coverage, -error, float(threshold), error))
    passing = [x for x in candidates if x[0]]
    chosen = max(passing or candidates, key=lambda x: (x[1] if x[0] else x[2], x[2]))
    class_thresholds = []
    for option in range(probability.shape[1]):
        rows = []
        for threshold in torch.linspace(0.5, 0.999, 101):
            selected = (prediction == option) & (confidence >= threshold)
            if int(selected.sum()) < 10:
                continue
            error = float((target[selected] != option).float().mean())
            coverage = float(selected.float().mean())
            rows.append((error <= 0.01, coverage, -error, float(threshold), error))
        passing_rows = [row for row in rows if row[0]]
        selected_row = max(passing_rows, key=lambda row: row[1]) if passing_rows else None
        class_thresholds.append(selected_row[3] if selected_row else 1.001)
    return {
        "threshold": chosen[3], "selective_error": chosen[4],
        "coverage": chosen[1], "class_thresholds_99_precision": class_thresholds,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="results/router/v4_option_prefixes_train_v1.npz")
    parser.add_argument("--metadata", default="results/router/v4_option_prefixes_train_v1.json")
    parser.add_argument("--output-dir", default="results/router/v4_causal_router_v1")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--models", nargs="+",
        choices=("causal_gru", "static_mlp", "unstructured_gru"),
        default=("causal_gru", "static_mlp", "unstructured_gru"),
    )
    args = parser.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raw = np.load(args.data)
    tensors = {key: torch.from_numpy(raw[key]) for key in raw.files}
    # Safe decision supervision: the target is DEFER until the causal prefix
    # contains enough post-onset evidence.  These cutoffs were frozen from the
    # data-collection envelope, not selected on control outcomes.
    tensors["option"], tensors["block_status"] = causal_safe_targets(tensors)
    train, validation, test = group_split(raw["group_id"])
    valid_steps = []
    for sequence, length in zip(raw["sequence"][train], raw["length"][train]):
        valid_steps.append(sequence[:length])
    valid_steps = np.concatenate(valid_steps)
    mean = torch.from_numpy(valid_steps.mean(0)).float()
    scale = torch.from_numpy(valid_steps.std(0)).float().clamp_min(1e-5)
    train_indices = torch.from_numpy(np.flatnonzero(train))
    train_data = TensorDataset(
        tensors["sequence"][train_indices], tensors["length"][train_indices],
        tensors["option"][train_indices], tensors["event"][train_indices],
        tensors["direction"][train_indices], tensors["block_status"][train_indices],
    )
    output_dir = Path(args.output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    summary = {"schema_version": 1, "seed": args.seed, "models": {}}
    for name in args.models:
        torch.manual_seed(args.seed)
        model = make_model(name, raw["sequence"].shape[-1], args.hidden_dim).to(device)
        model.set_normalization(mean.to(device), scale.to(device))
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
        loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, generator=torch.Generator().manual_seed(args.seed))
        best = None
        for epoch in range(args.epochs):
            model.train()
            for batch in loader:
                batch = tuple(item.to(device) for item in batch)
                optimizer.zero_grad(set_to_none=True)
                loss = loss_for(model, batch, name != "unstructured_gru")
                loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); optimizer.step()
            validation_metrics, validation_probability, validation_target = evaluate(model, tensors, validation, device)
            candidate = (validation_metrics["nll"], epoch, {k: v.detach().cpu() for k, v in model.state_dict().items()})
            if best is None or candidate[0] < best[0]: best = candidate
        model.load_state_dict(best[2])
        validation_metrics, validation_probability, validation_target = evaluate(model, tensors, validation, device)
        calibration = calibrate(validation_probability, validation_target)
        test_metrics, test_probability, test_target = evaluate(model, tensors, test, device)
        confidence, prediction = test_probability.max(1)
        selected = confidence >= calibration["threshold"]
        test_metrics["selective"] = {
            "threshold_from_validation": calibration["threshold"],
            "coverage": float(selected.float().mean()),
            "error": float((prediction[selected] != test_target[selected]).float().mean()) if bool(selected.any()) else None,
        }
        checkpoint = {
            "schema_version": 1, "model": name, "seed": args.seed,
            "input_dim": int(raw["sequence"].shape[-1]), "hidden_dim": args.hidden_dim,
            "state_dict": model.state_dict(), "calibration": calibration,
            "data_sha256": hashlib.sha256(Path(args.data).read_bytes()).hexdigest(),
            "feature_metadata_sha256": hashlib.sha256(Path(args.metadata).read_bytes()).hexdigest(),
            "best_epoch": best[1],
        }
        path = output_dir / f"{name}_seed{args.seed}.pt"
        torch.save(checkpoint, path)
        summary["models"][name] = {
            "best_epoch": best[1], "validation": validation_metrics,
            "calibration": calibration, "test": test_metrics,
            "checkpoint": str(path),
            "checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    summary["split_rows"] = {"train": int(train.sum()), "validation": int(validation.sum()), "test": int(test.sum())}
    result_path = output_dir / f"summary_seed{args.seed}.json"
    result_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

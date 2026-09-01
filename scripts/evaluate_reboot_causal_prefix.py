#!/usr/bin/env python3
"""Leave-one-object-out REBOOT evaluation with matched proprioceptive inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class StaticMLP(nn.Module):
    def __init__(self, width, hidden=96):
        super().__init__(); self.net = nn.Sequential(nn.Linear(width, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 1))
    def forward(self, sequence, length):
        index = length - 1
        final = sequence[torch.arange(len(sequence), device=sequence.device), index]
        return self.net(final).squeeze(1), None


class MomentMLP(nn.Module):
    def __init__(self, width, hidden=96):
        super().__init__(); self.net = nn.Sequential(nn.Linear(2 * width, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 1))
    def forward(self, sequence, length):
        time = torch.arange(sequence.shape[1], device=sequence.device)[None] < length[:, None]
        count = length.float()[:, None]
        mean = (sequence * time[:, :, None]).sum(1) / count
        variance = ((sequence - mean[:, None]).square() * time[:, :, None]).sum(1) / count
        return self.net(torch.cat((mean, variance.sqrt()), dim=1)).squeeze(1), None


class EndpointPairMLP(nn.Module):
    """Ladder rung 2: the current frame plus one earlier frame, no sequence model.

    REBOOT prefixes are not current-centered, so unlike LearnedRecovery-v4 the
    final frame alone already carries signal (that is `StaticMLP`). The
    matching rung-2 control here is therefore the *pair* of endpoints: what a
    model can tell from where the trajectory started and where it is now,
    without any temporal encoder or summary over the frames between.
    """

    def __init__(self, width, hidden=96):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * width, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 1),
        )

    def forward(self, sequence, length):
        index = torch.arange(len(sequence), device=sequence.device)
        final = sequence[index, length - 1]
        first = sequence[:, 0]
        return self.net(torch.cat((first, final), dim=1)).squeeze(1), None


class TemporalGRU(nn.Module):
    def __init__(self, width, hidden=96, dynamics=False):
        super().__init__(); self.dynamics = dynamics
        self.gru = nn.GRU(width, hidden, 2, batch_first=True, dropout=0.1)
        self.head = nn.Linear(hidden, 1)
        self.dynamics_head = nn.Linear(hidden, 14) if dynamics else None
    def forward(self, sequence, length):
        encoded, _ = self.gru(sequence)
        latent = encoded[torch.arange(len(sequence), device=sequence.device), length - 1]
        return self.head(latent).squeeze(1), self.dynamics_head(latent) if self.dynamics else None


def make_model(name, width):
    if name == "static_mlp": return StaticMLP(width)
    if name == "moment_mlp": return MomentMLP(width)
    if name == "endpoint_pair_mlp": return EndpointPairMLP(width)
    if name == "unstructured_gru": return TemporalGRU(width, dynamics=False)
    if name == "causal_dynamics_gru": return TemporalGRU(width, dynamics=True)
    raise ValueError(name)


def batches(indices, sequence, label, batch_size, shuffle, seed):
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(TensorDataset(torch.from_numpy(indices), torch.from_numpy(label[indices])), batch_size=batch_size, shuffle=shuffle, generator=generator)


def normalize(sequence, train_mask):
    rows = sequence[train_mask].reshape(-1, sequence.shape[-1]).astype(np.float64)
    return rows.mean(0).astype(np.float32), rows.std(0).clip(1e-5).astype(np.float32)


@torch.inference_mode()
def predict(model, sequence, indices, horizon, mean, scale, device):
    result = []
    loader = DataLoader(TensorDataset(torch.from_numpy(indices)), batch_size=128)
    model.eval()
    for (index,) in loader:
        x = torch.from_numpy(sequence[index.numpy(), :horizon]).to(device)
        x = (x - mean) / scale
        length = torch.full((len(x),), horizon, dtype=torch.long, device=device)
        logit, _ = model(x, length); result.append(logit.sigmoid().cpu().numpy())
    return np.concatenate(result)


def fit_one(name, sequence, label, train_mask, validation_mask, seed, args, device):
    mean_np, scale_np = normalize(sequence, train_mask)
    mean = torch.from_numpy(mean_np).to(device); scale = torch.from_numpy(scale_np).to(device)
    # Seed per (method, fold) before constructing the model. Without this, weight
    # initialisation draws from the global RNG, so adding or reordering a method
    # silently changes the initialisation of every method after it and the arms
    # stop being comparable.
    torch.manual_seed(hash((name, seed)) % (2**31))
    model = make_model(name, sequence.shape[-1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    train_indices = np.flatnonzero(train_mask)
    positive = label[train_indices].sum(); negative = len(train_indices) - positive
    positive_weight = torch.tensor(negative / max(positive, 1), device=device)
    best = None
    for epoch in range(args.epochs):
        model.train()
        loader = batches(train_indices, sequence, label, args.batch_size, True, seed * 1000 + epoch)
        for index, target in loader:
            index_np = index.numpy()
            horizon_choices = np.asarray(args.horizons)
            # One causal cutoff per minibatch avoids padding as an unintended cue.
            horizon = int(horizon_choices[(epoch + int(index_np[0])) % len(horizon_choices)])
            x = torch.from_numpy(sequence[index_np, :horizon]).to(device)
            x = (x - mean) / scale
            length = torch.full((len(x),), horizon, dtype=torch.long, device=device)
            target = target.float().to(device)
            logit, dynamics = model(x, length)
            loss = nn.functional.binary_cross_entropy_with_logits(logit, target, pos_weight=positive_weight)
            if dynamics is not None and horizon < sequence.shape[1]:
                future = torch.from_numpy(sequence[index_np, horizon:min(horizon + 32, sequence.shape[1]), -14:].mean(1)).to(device)
                # Predict in the same train-fold coordinate system used by the
                # encoder. Omitting the training mean here biases every
                # dynamics target and makes the auxiliary task object-specific.
                future_mean = mean[-14:]
                future_scale = scale[-14:]
                normalized_future = (future - future_mean) / future_scale
                loss = loss + 0.2 * nn.functional.smooth_l1_loss(dynamics, normalized_future)
            optimizer.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 5); optimizer.step()
        validation_indices = np.flatnonzero(validation_mask)
        probabilities = predict(model, sequence, validation_indices, max(args.horizons), mean, scale, device)
        score = roc_auc_score(label[validation_indices], probabilities)
        candidate = (score, epoch, {k: v.detach().cpu() for k, v in model.state_dict().items()})
        if best is None or candidate[0] > best[0]: best = candidate
    model.load_state_dict(best[2])
    return model, mean, scale, best[1]


def metrics(target, probability):
    return {
        "episodes": int(len(target)), "auroc": float(roc_auc_score(target, probability)),
        "balanced_accuracy": float(balanced_accuracy_score(target, probability >= 0.5)),
        "positive_rate": float(target.mean()),
    }


def bootstrap_difference(folds, candidate, baseline, seed=20260831):
    rng = np.random.default_rng(seed); differences = []
    values = np.asarray([fold[candidate]["auroc"] - fold[baseline]["auroc"] for fold in folds])
    for _ in range(10000): differences.append(rng.choice(values, size=len(values), replace=True).mean())
    return [float(np.quantile(differences, 0.025)), float(np.quantile(differences, 0.975))]


def stratified_subsample(mask, object_id, label, fraction, seed):
    if fraction >= 1.0:
        return mask
    rng = np.random.default_rng(seed)
    selected = np.zeros_like(mask, dtype=bool)
    for object_value in np.unique(object_id[mask]):
        for label_value in (0, 1):
            indices = np.flatnonzero(mask & (object_id == object_value) & (label == label_value))
            count = max(2, int(np.ceil(len(indices) * fraction)))
            selected[rng.choice(indices, size=min(count, len(indices)), replace=False)] = True
    return selected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="results/reboot/reboot_prefix_v1.npz")
    parser.add_argument("--audit", default="results/reboot/reboot_prefix_v1.audit.json")
    parser.add_argument("--output", default="results/reboot/reboot_causal_prefix_seed0.json")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--horizons", type=int, nargs="+", default=[64, 128, 256])
    parser.add_argument("--train-fraction", type=float, default=1.0)
    args = parser.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raw = np.load(args.data); sequence = raw["sequence"]; label = raw["label"]; object_id = raw["object_id"]
    audit = json.loads(Path(args.audit).read_text()); object_names = audit["objects"]
    methods = (
        "static_mlp", "endpoint_pair_mlp", "moment_mlp",
        "unstructured_gru", "causal_dynamics_gru",
    )
    fold_results = []
    for test_object in sorted(np.unique(object_id)):
        validation_object = (test_object + 1) % len(object_names)
        train_mask = (object_id != test_object) & (object_id != validation_object)
        train_mask = stratified_subsample(
            train_mask, object_id, label, args.train_fraction,
            args.seed * 10_000 + int(test_object),
        )
        validation_mask = object_id == validation_object
        test_mask = object_id == test_object
        fold = {"test_object": object_names[test_object], "validation_object": object_names[validation_object]}
        for name in methods:
            model, mean, scale, epoch = fit_one(name, sequence, label, train_mask, validation_mask, args.seed, args, device)
            fold[name] = {"best_epoch": epoch, "by_horizon": {}}
            for horizon in args.horizons:
                index = np.flatnonzero(test_mask)
                probability = predict(model, sequence, index, horizon, mean, scale, device)
                fold[name]["by_horizon"][str(horizon)] = metrics(label[index], probability)
            fold[name].update(fold[name]["by_horizon"][str(max(args.horizons))])
        fold_results.append(fold)
    aggregate = {}
    for name in methods:
        aggregate[name] = {
            metric: float(np.mean([fold[name][metric] for fold in fold_results]))
            for metric in ("auroc", "balanced_accuracy")
        }
    aggregate["causal_vs_static_auroc_difference"] = aggregate["causal_dynamics_gru"]["auroc"] - aggregate["static_mlp"]["auroc"]
    aggregate["causal_vs_static_object_bootstrap_95"] = bootstrap_difference(fold_results, "causal_dynamics_gru", "static_mlp")
    aggregate["causal_vs_moment_auroc_difference"] = aggregate["causal_dynamics_gru"]["auroc"] - aggregate["moment_mlp"]["auroc"]
    aggregate["causal_vs_moment_object_bootstrap_95"] = bootstrap_difference(fold_results, "causal_dynamics_gru", "moment_mlp")
    aggregate["causal_vs_unstructured_auroc_difference"] = aggregate["causal_dynamics_gru"]["auroc"] - aggregate["unstructured_gru"]["auroc"]
    aggregate["causal_vs_unstructured_object_bootstrap_95"] = bootstrap_difference(fold_results, "causal_dynamics_gru", "unstructured_gru")
    payload = {
        "schema_version": 1, "seed": args.seed, "protocol": "leave-one-object-out",
        "train_fraction": args.train_fraction,
        "horizons": args.horizons, "folds": fold_results, "aggregate": aggregate,
        "data_sha256": hashlib.sha256(Path(args.data).read_bytes()).hexdigest(),
        "audit_sha256": hashlib.sha256(Path(args.audit).read_bytes()).hexdigest(),
        "claim_boundary": "Offline real-robot recovery-prefix classification, not closed-loop control.",
    }
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(aggregate, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

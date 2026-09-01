#!/usr/bin/env python3
"""Score a benchmark's held-out mechanism against a ladder of control models.

A recovery benchmark is only measuring recovery if its held-out mechanism
cannot be identified by something trivial. This audit runs four rungs of
increasing capability against the identical matched tensor, group-disjoint
split, and held-out option:

  1. instantaneous  -- the current frame only. Under current-centering this is
                       exactly zero, so it is a contract check, not a model.
  2. one past frame -- a single earlier observation, no sequence model. Under
                       current-centering that frame carries signed displacement
                       to the present.
  3. hand-written   -- a motion-threshold rule with no learned parameters.
  4. recurrent      -- the factorized and unstructured sequence models.

If a lower rung matches the top rung on the held-out mechanism, that mechanism
is a shortcut and cannot support a composition claim, regardless of how large
the margin over a weak learned baseline looks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from atr.policies.option_router import (
    FactorizedOptionRouter, MomentSummaryRouter, StaticOffsetRouter, StaticOptionRouter,
    UnstructuredOptionGRU, current_centered_sequence, deployable_option_targets,
)
from atr.policies.heuristic_option_router import (
    HeuristicMotionRouter, SignedAxisMotionRouter,
)

# Rung 2b is order-free but reads the whole prefix. It is the strongest
# non-recurrent control and the one that decides whether a verdict is robust:
# a weak rung-2 can make any benchmark look shortcut-free.
RUNG = {
    "instantaneous": 1, "one_past_frame": 2, "moment_summary": 2.5,
    "hand_written": 3, "recurrent": 4,
}


def group_bootstrap_difference(
    top_rows, top_groups, low_rows, low_groups, seed=20260901, samples=10000,
):
    """Bootstrap rung4 minus a lower rung, resampling whole episode groups.

    Prefixes from one episode are correlated, so the episode is the resampling
    unit. Both rungs are scored on the same rows, so the paired difference is
    taken within each resampled group.
    """
    import collections

    top_by_group = collections.defaultdict(list)
    low_by_group = collections.defaultdict(list)
    for hit, group in zip(top_rows, top_groups):
        top_by_group[group].append(hit)
    for hit, group in zip(low_rows, low_groups):
        low_by_group[group].append(hit)
    shared = sorted(set(top_by_group) & set(low_by_group))
    if not shared:
        return None
    top_mean = np.array([np.mean(top_by_group[g]) for g in shared])
    low_mean = np.array([np.mean(low_by_group[g]) for g in shared])
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(shared), size=(samples, len(shared)))
    differences = top_mean[draws].mean(1) - low_mean[draws].mean(1)
    return {
        "difference": float(top_mean.mean() - low_mean.mean()),
        "group_bootstrap_95": [
            float(np.quantile(differences, 0.025)),
            float(np.quantile(differences, 0.975)),
        ],
        "groups": len(shared),
    }


def group_split(group_id: np.ndarray):
    bucket = np.array([
        int(hashlib.sha256(str(int(x)).encode()).hexdigest()[:8], 16) % 100
        for x in group_id
    ])
    return bucket < 70, (bucket >= 70) & (bucket < 85), bucket >= 85


@torch.inference_mode()
def score(model, tensors, indices, geometry_dim, heldout_option, device, groups, batch=512):
    """Held-out-option accuracy and overall accuracy on the given rows."""
    loader = DataLoader(TensorDataset(torch.from_numpy(indices)), batch_size=batch)
    correct = total = heldout_correct = heldout_total = 0
    heldout_rows, heldout_groups = [], []
    for (index,) in loader:
        sequence = tensors["sequence"][index].to(device)
        length = tensors["length"][index].to(device)
        target = tensors["option"][index].to(device)
        sequence = current_centered_sequence(sequence, length, geometry_dim)
        output = model(sequence, length)
        logp = (
            output.option_log_probability
            if hasattr(output, "option_log_probability") else output
        )
        prediction = logp.argmax(-1)
        correct += int((prediction == target).sum())
        total += int(target.numel())
        mask = target == heldout_option
        if bool(mask.any()):
            hit = (prediction[mask] == target[mask])
            heldout_correct += int(hit.sum())
            heldout_total += int(mask.sum())
            heldout_rows.extend(hit.int().cpu().tolist())
            heldout_groups.extend(groups[index.numpy()][mask.cpu().numpy()].tolist())
    return {
        "rows": total,
        "accuracy": correct / total if total else None,
        "heldout_rows": heldout_total,
        "heldout_option_accuracy": (
            heldout_correct / heldout_total if heldout_total else None
        ),
        "heldout_correct_by_row": heldout_rows,
        "heldout_group_by_row": heldout_groups,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=(0, 1, 2))
    parser.add_argument(
        "--heuristic", choices=("actor_pair", "signed_axis"), default="actor_pair",
        help="actor_pair for LearnedRecovery-v4, signed_axis for PegInsertion.",
    )
    parser.add_argument("--physical-heldout-only", action="store_true")
    parser.add_argument(
        "--geometry-dim", type=int,
        help="Override; older metadata files omit current_centered_geometry_dim.",
    )
    parser.add_argument("--heldout-option", type=int, help="Override for older metadata.")
    args = parser.parse_args()

    metadata = json.loads(args.metadata.read_text())
    feature_names = metadata["feature_names"]
    geometry_dim = (
        args.geometry_dim if args.geometry_dim is not None
        else int(metadata["current_centered_geometry_dim"])
    )
    heldout_option = (
        args.heldout_option if args.heldout_option is not None
        else int(metadata["heldout_option"])
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    raw = np.load(args.data)
    tensors = {key: torch.from_numpy(raw[key]) for key in raw.files}
    tensors["option"], _ = deployable_option_targets(tensors)
    _, _, test = group_split(raw["group_id"])
    if args.physical_heldout_only and "physical_heldout" in raw.files:
        # Restrict held-out scoring to genuinely observed prefixes rather than
        # counterfactually reflected ones.
        test = test & (raw["physical_heldout"].astype(bool) | (
            tensors["option"].numpy() != heldout_option
        ))
    indices = np.flatnonzero(test)

    report = {
        "schema_version": 1,
        "data_sha256": hashlib.sha256(args.data.read_bytes()).hexdigest(),
        "feature_metadata_sha256": hashlib.sha256(args.metadata.read_bytes()).hexdigest(),
        "heldout_option": heldout_option,
        "geometry_dim": geometry_dim,
        "physical_heldout_only": bool(args.physical_heldout_only),
        "rungs": {},
    }

    def record(name, rung, results):
        values = [r["heldout_option_accuracy"] for r in results
                  if r["heldout_option_accuracy"] is not None]
        overall = [r["accuracy"] for r in results if r["accuracy"] is not None]
        report["rungs"][name] = {
            "rung": rung,
            "seeds": results,
            "heldout_option_accuracy_mean": (
                sum(values) / len(values) if values else None
            ),
            "accuracy_mean": sum(overall) / len(overall) if overall else None,
        }

    # Rung 3: hand-written, no parameters and no seed dependence.
    if args.heuristic == "signed_axis":
        heuristic = SignedAxisMotionRouter(feature_names).to(device).eval()
    else:
        heuristic = HeuristicMotionRouter(feature_names).to(device).eval()
    record("hand_written", RUNG["hand_written"],
           [score(heuristic, tensors, indices, geometry_dim, heldout_option, device, raw['group_id'])])

    # Rungs 1, 2 and 4: learned models, one checkpoint per seed.
    families = {
        "instantaneous": ("static_mlp", StaticOptionRouter),
        "one_past_frame": ("static_offset_first", StaticOffsetRouter),
        "moment_summary": ("moment_summary", MomentSummaryRouter),
        "recurrent_factorized": ("causal_gru", FactorizedOptionRouter),
        "recurrent_unstructured": ("unstructured_gru", UnstructuredOptionGRU),
    }
    for name, (stem, factory) in families.items():
        results = []
        for seed in args.seeds:
            path = args.checkpoint_dir / f"{stem}_seed{seed}.pt"
            if not path.exists():
                continue
            checkpoint = torch.load(path, map_location=device, weights_only=False)
            if factory is MomentSummaryRouter:
                model = factory(checkpoint["input_dim"], checkpoint["hidden_dim"])
            elif factory is StaticOffsetRouter:
                model = factory(checkpoint["input_dim"], checkpoint["hidden_dim"], None)
            elif factory is StaticOptionRouter:
                model = factory(checkpoint["input_dim"], checkpoint["hidden_dim"])
            else:
                model = factory(checkpoint["input_dim"], checkpoint["hidden_dim"], 2)
            model.load_state_dict(checkpoint["state_dict"])
            model.to(device).eval()
            results.append(
                score(model, tensors, indices, geometry_dim, heldout_option, device, raw['group_id'])
            )
        if results:
            rung = RUNG["recurrent"] if name.startswith("recurrent") else RUNG[name]
            record(name, rung, results)

    # Statistical criterion: a lower rung is "matching" when the recurrent
    # model's advantage over it is not distinguishable from zero, i.e. the
    # paired group-bootstrap interval on (rung4 - lower) includes zero. This
    # replaces an arbitrary ratio cut with a test that has an error rate.
    top_entry = report["rungs"].get("recurrent_factorized")
    top = top_entry["heldout_option_accuracy_mean"] if top_entry else None
    comparisons, flagged = {}, []
    if top_entry:
        top_seed = top_entry["seeds"][0]
        for name, entry in report["rungs"].items():
            if entry["rung"] >= RUNG["recurrent"]:
                continue
            low_seed = entry["seeds"][0]
            test = group_bootstrap_difference(
                top_seed["heldout_correct_by_row"], top_seed["heldout_group_by_row"],
                low_seed["heldout_correct_by_row"], low_seed["heldout_group_by_row"],
            )
            if test is None:
                continue
            test["indistinguishable_from_rung4"] = bool(test["group_bootstrap_95"][0] <= 0)
            test["ratio_to_rung4"] = (
                entry["heldout_option_accuracy_mean"] / top if top else None
            )
            comparisons[name] = test
            if test["indistinguishable_from_rung4"]:
                flagged.append(name)

    report["comparisons_to_rung4"] = comparisons
    report["verdict"] = {
        "criterion": (
            "A lower rung matches rung 4 when the paired group-bootstrap 95% "
            "interval on (rung4 - lower) includes zero."
        ),
        "top_rung_heldout_accuracy": top,
        "matching_lower_rungs": flagged,
        "heldout_mechanism_is_a_shortcut": bool(flagged) if top_entry else None,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")

    print(f"{'rung':>5}  {'control':<24} {'held-out acc':>13}  {'overall':>8}")
    print("-" * 56)
    for name, entry in sorted(report["rungs"].items(), key=lambda kv: kv[1]["rung"]):
        held = entry["heldout_option_accuracy_mean"]
        overall = entry["accuracy_mean"]
        held_s = "n/a" if held is None else f"{held:.4f}"
        overall_s = "n/a" if overall is None else f"{overall:.4f}"
        print(f"{entry['rung']:>5}  {name:<24} {held_s:>13}  {overall_s:>8}")
    for name, test in report["comparisons_to_rung4"].items():
        lo, hi = test["group_bootstrap_95"]
        mark = "MATCHES rung 4" if test["indistinguishable_from_rung4"] else ""
        print(f"  rung4 - {name:<24} {test['difference']:+.4f} "
              f"[{lo:+.4f}, {hi:+.4f}]  {mark}")
    verdict = report["verdict"]["heldout_mechanism_is_a_shortcut"]
    print(f"\nheld-out mechanism is a shortcut: {verdict} "
          f"(matching rungs: {report['verdict']['matching_lower_rungs'] or 'none'})")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()

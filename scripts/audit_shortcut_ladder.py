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
    FactorizedOptionRouter, StaticOffsetRouter, StaticOptionRouter,
    UnstructuredOptionGRU, current_centered_sequence, deployable_option_targets,
)
from atr.policies.heuristic_option_router import (
    HeuristicMotionRouter, SignedAxisMotionRouter,
)

RUNG = {
    "instantaneous": 1, "one_past_frame": 2, "hand_written": 3, "recurrent": 4,
}


def group_split(group_id: np.ndarray):
    bucket = np.array([
        int(hashlib.sha256(str(int(x)).encode()).hexdigest()[:8], 16) % 100
        for x in group_id
    ])
    return bucket < 70, (bucket >= 70) & (bucket < 85), bucket >= 85


@torch.inference_mode()
def score(model, tensors, indices, geometry_dim, heldout_option, device, batch=512):
    """Held-out-option accuracy and overall accuracy on the given rows."""
    loader = DataLoader(TensorDataset(torch.from_numpy(indices)), batch_size=batch)
    correct = total = heldout_correct = heldout_total = 0
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
            heldout_correct += int((prediction[mask] == target[mask]).sum())
            heldout_total += int(mask.sum())
    return {
        "rows": total,
        "accuracy": correct / total if total else None,
        "heldout_rows": heldout_total,
        "heldout_option_accuracy": (
            heldout_correct / heldout_total if heldout_total else None
        ),
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
           [score(heuristic, tensors, indices, geometry_dim, heldout_option, device)])

    # Rungs 1, 2 and 4: learned models, one checkpoint per seed.
    families = {
        "instantaneous": ("static_mlp", StaticOptionRouter),
        "one_past_frame": ("static_offset_first", StaticOffsetRouter),
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
            if factory is StaticOffsetRouter:
                model = factory(checkpoint["input_dim"], checkpoint["hidden_dim"], None)
            elif factory is StaticOptionRouter:
                model = factory(checkpoint["input_dim"], checkpoint["hidden_dim"])
            else:
                model = factory(checkpoint["input_dim"], checkpoint["hidden_dim"], 2)
            model.load_state_dict(checkpoint["state_dict"])
            model.to(device).eval()
            results.append(
                score(model, tensors, indices, geometry_dim, heldout_option, device)
            )
        if results:
            rung = RUNG["recurrent"] if name.startswith("recurrent") else RUNG[name]
            record(name, rung, results)

    top = report["rungs"].get("recurrent_factorized", {}).get(
        "heldout_option_accuracy_mean"
    )
    lower = {
        name: entry["heldout_option_accuracy_mean"]
        for name, entry in report["rungs"].items()
        if entry["rung"] < RUNG["recurrent"]
        and entry["heldout_option_accuracy_mean"] is not None
    }
    best_lower = max(lower.values()) if lower else None
    report["verdict"] = {
        "top_rung_heldout_accuracy": top,
        "best_lower_rung": (
            max(lower, key=lower.get) if lower else None
        ),
        "best_lower_rung_heldout_accuracy": best_lower,
        "heldout_mechanism_is_a_shortcut": (
            None if (top is None or best_lower is None) else bool(best_lower >= 0.9 * top)
        ),
        "interpretation": (
            "A lower rung matching the recurrent model means the held-out "
            "mechanism is identifiable without the capability under test."
        ),
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
    verdict = report["verdict"]["heldout_mechanism_is_a_shortcut"]
    print(f"\nheld-out mechanism is a shortcut: {verdict}")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()

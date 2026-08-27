#!/usr/bin/env python3
"""Summarize learned and scripted high-level policies on the held-out split.

These policies use the abstract high-level skill executor and are explicitly
reported as decision-layer diagnostics, not physical manipulation results.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np


def _ci(values, seed=20260826, samples=10000):
    values = np.asarray(values, dtype=float)
    if len(values) == 1:
        return [float(values[0]), float(values[0])]
    rng = np.random.default_rng(seed)
    means = np.mean(rng.choice(values, size=(samples, len(values)), replace=True), axis=1)
    return [float(x) for x in np.quantile(means, [0.025, 0.975])]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="results/rl_training")
    args = parser.parse_args()
    root = Path(args.root)
    specifications = [
        ("adaptive_recovery_rl_v1", "feasibility_q"),
        ("adaptive_recovery_rl_v1", "privileged_mechanism_q"),
        ("adaptive_recovery_rl_baselines_v1", "blind_domain_randomized_q"),
        ("adaptive_recovery_rl_baselines_v1", "behavioral_cloning"),
    ]
    summaries = []
    for experiment, variant in specifications:
        paths = sorted((root / experiment / variant).glob("seed_*/result.json"))
        if len(paths) != 10:
            raise RuntimeError(f"expected 10 results for {variant}, found {len(paths)}")
        results = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
        seed_rows = []
        for result in results:
            episodes = result["test"]["episodes"]
            seed_rows.append({
                "score": float(np.mean([row["score"] for row in episodes])),
                "goals_achieved": float(np.mean([row["goals_achieved"] for row in episodes])),
                "wasted_steps": float(np.mean([row["wasted_steps"] for row in episodes])),
            })
        summary = {"policy": variant, "training_seeds": len(seed_rows), "test_episodes": sum(len(r["test"]["episodes"]) for r in results)}
        for metric in ("score", "goals_achieved", "wasted_steps"):
            values = [row[metric] for row in seed_rows]
            summary[f"mean_{metric}"] = float(np.mean(values))
            summary[f"{metric}_bootstrap_95"] = _ci(values)
        summaries.append(summary)

    scripted = json.loads((root / "scripted_baselines_v1.json").read_text(encoding="utf-8"))
    for policy in sorted({row["policy"] for row in scripted["records"]}):
        rows = [row for row in scripted["records"] if row["policy"] == policy]
        summary = {"policy": policy, "training_seeds": 0, "test_episodes": len(rows)}
        for metric in ("score", "goals_achieved", "wasted_steps", "constraint_violations"):
            values = [row[metric] for row in rows]
            summary[f"mean_{metric}"] = float(np.mean(values))
            summary[f"{metric}_bootstrap_95"] = _ci(values)
        summaries.append(summary)

    payload = {
        "schema_version": 1,
        "scope": "high-level abstract-skill diagnostic; not physical manipulation",
        "test_split": "held_out_intervention",
        "policies": summaries,
    }
    output = root / "aggregate_v1.json"
    temporary = output.with_name(f".{output.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, output)
    csv_path = root / "summary_v1.csv"
    fields = sorted({key for row in summaries for key in row if not key.endswith("_95")})
    temporary_csv = csv_path.with_name(f".{csv_path.name}.tmp.{os.getpid()}")
    with temporary_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in summaries:
            writer.writerow({key: row.get(key, "") for key in fields})
    os.replace(temporary_csv, csv_path)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

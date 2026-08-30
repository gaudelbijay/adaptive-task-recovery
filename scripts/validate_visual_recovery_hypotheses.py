#!/usr/bin/env python3
"""Compute auditable V1--V5 verdicts from frozen held-out aggregates."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from aggregate_visual_recovery import paired_effect
from compare_visual_recovery_candidates import paired_seed_groups, state_method, visual_method


def atomic_text(text, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def load_payload(path, semantics, allow_missing=False):
    path = Path(path)
    if not path.exists():
        if allow_missing:
            return None
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("benchmark_semantics") != semantics:
        raise ValueError(f"aggregate semantics mismatch: {path}")
    return payload


def require_protocol(result, seeds, episodes):
    if int(result.get("seeds", -1)) != seeds:
        raise ValueError("hypothesis input has the wrong training-seed count")
    if int(result.get("episodes", -1)) != episodes:
        raise ValueError("hypothesis input has the wrong held-out episode count")
    return result


def load_strict_payload(path, semantics, protocol, allow_missing=False):
    payload = load_payload(path, semantics, allow_missing)
    if payload is not None and payload.get("protocol") != protocol:
        raise ValueError(f"strict aggregate protocol mismatch: {path}")
    return payload


def strict_cohort(payload, label, seeds, episodes):
    matches = [item for item in payload["cohorts"] if item["label"] == label]
    if len(matches) != 1:
        raise ValueError(f"strict aggregate lacks unique cohort: {label}")
    result = matches[0]
    if len(result.get("training_seeds", [])) != seeds:
        raise ValueError("strict cohort has the wrong training-seed count")
    if int(result.get("episodes", -1)) != episodes:
        raise ValueError("strict cohort has the wrong held-out episode count")
    return result


def strict_comparison(payload, left, right, branch=None):
    comparisons = (
        payload["paired_comparisons"] if branch is None
        else payload["paired_comparisons_by_branch"][branch]
    )
    matches = [
        item for item in comparisons
        if {item["left"], item["right"]} == {left, right}
    ]
    if len(matches) != 1:
        raise ValueError(f"strict aggregate lacks unique comparison: {left}, {right}")
    result = dict(matches[0])
    if result["left"] != left:
        for key in ("success_rate_difference", "safe_success_rate_difference"):
            result[key] = -result[key]
        for key in ("paired_bootstrap_95", "safe_paired_bootstrap_95"):
            result[key] = [-result[key][1], -result[key][0]]
        result["left"], result["right"] = left, right
    return result


def paired_visual(left, right, branch=None, branch_value=None):
    left_groups, right_groups = paired_seed_groups(left, right)
    if branch is not None:
        filtered_left, filtered_right = [], []
        for left_seed, right_seed in zip(left_groups, right_groups):
            pairs = [
                (a, b) for a, b in zip(left_seed, right_seed)
                if a.get(branch) == branch_value and b.get(branch) == branch_value
            ]
            if not pairs:
                raise ValueError(f"paired branch has no episodes: {branch}={branch_value}")
            filtered_left.append([item[0] for item in pairs])
            filtered_right.append([item[1] for item in pairs])
        left_groups, right_groups = filtered_left, filtered_right
    return paired_effect(left_groups, right_groups, np.random.default_rng(20260828))


def verdict(passed):
    if passed is None:
        return "pending"
    return "confirmed" if passed else "rejected"


def validate(config, allow_missing=False):
    semantics = config["benchmark_semantics"]
    seeds = int(config["required_training_seeds"])
    episodes = int(config["required_episodes"])
    output = {
        "schema_version": 1,
        "protocol": "V1--V5 held-out hypothesis validation",
        "benchmark_semantics": semantics,
        "required_training_seeds": seeds,
        "required_episodes": episodes,
        "hypotheses": {},
    }

    v1_cfg = config["v1"]
    v1_payload = load_payload(v1_cfg["primary"]["path"], semantics, allow_missing)
    if v1_payload is None:
        output["hypotheses"]["V1"] = {"verdict": "pending", "primary": True}
    else:
        v1 = require_protocol(
            visual_method(v1_payload, "nominal", v1_cfg["primary"]["method"]),
            seeds, episodes,
        )
        passed = float(v1["success_rate"]) >= float(v1_cfg["minimum_nominal_success"])
        output["hypotheses"]["V1"] = {
            "verdict": verdict(passed), "primary": True,
            "success_rate": v1["success_rate"],
            "safe_success_rate": v1["safe_success_rate"],
            "hierarchical_bootstrap_95": v1["success_hierarchical_bootstrap_95"],
            "minimum_success_rate": v1_cfg["minimum_nominal_success"],
            "method": v1_cfg["primary"]["method"],
            "disclosure": v1_cfg["primary"]["disclosure"],
        }
    fallback_cfg = v1_cfg.get("fallback_visual_competence")
    if fallback_cfg:
        fallback_payload = load_payload(fallback_cfg["path"], semantics, allow_missing)
        if fallback_payload is None:
            fallback_result = {"verdict": "pending", "primary": False}
        else:
            fallback = require_protocol(
                visual_method(fallback_payload, "nominal", fallback_cfg["method"]),
                seeds, episodes,
            )
            fallback_passed = float(fallback["success_rate"]) >= float(
                v1_cfg["minimum_nominal_success"]
            )
            fallback_result = {
                "verdict": verdict(fallback_passed), "primary": False,
                "success_rate": fallback["success_rate"],
                "safe_success_rate": fallback["safe_success_rate"],
                "hierarchical_bootstrap_95": fallback[
                    "success_hierarchical_bootstrap_95"
                ],
                "minimum_success_rate": v1_cfg["minimum_nominal_success"],
                "method": fallback_cfg["method"],
                "disclosure": fallback_cfg["disclosure"],
                "claim_boundary": "fallback competence cannot overturn the primary V1 verdict",
            }
        output["hypotheses"]["V1"]["fallback_visual_competence"] = fallback_result

    v2_cfg = config["v2"]
    v2_payload = load_payload(v2_cfg["path"], semantics, allow_missing)
    if v2_payload is None:
        output["hypotheses"]["V2"] = {"verdict": "pending", "primary": True}
    else:
        treatment = require_protocol(
            visual_method(v2_payload, "nominal", v2_cfg["primary"]["treatment"]),
            seeds, episodes,
        )
        control = require_protocol(
            visual_method(v2_payload, "nominal", v2_cfg["primary"]["control"]),
            seeds, episodes,
        )
        effect = paired_visual(treatment, control)
        passed = effect["paired_bootstrap_95"][0] > 0
        output["hypotheses"]["V2"] = {
            "verdict": verdict(passed), "primary": True,
            "treatment": v2_cfg["primary"]["treatment"],
            "control": v2_cfg["primary"]["control"],
            "paired_effect": effect,
        }

    v3_cfg = config["v3"]
    v3_results = []
    for label in ("primary", "protocol_extension"):
        comparison = v3_cfg[label]
        payload = load_payload(comparison["path"], semantics, allow_missing)
        if payload is None:
            v3_results.append({"label": label, "verdict": "pending"})
            continue
        treatment = require_protocol(
            visual_method(payload, "nominal", comparison["treatment"]), seeds, episodes,
        )
        control = require_protocol(
            visual_method(payload, "nominal", comparison["control"]), seeds, episodes,
        )
        effect = paired_visual(treatment, control)
        pooled = float(treatment["success_rate"]) - float(control["success_rate"])
        passed = (
            pooled >= float(v3_cfg["minimum_pooled_improvement"])
            or effect["paired_bootstrap_95"][0] > 0
        )
        v3_results.append({
            "label": label, "verdict": verdict(passed),
            "treatment": comparison["treatment"], "control": comparison["control"],
            "pooled_success_difference": pooled, "paired_effect": effect,
        })
    primary_v3 = next(item for item in v3_results if item["label"] == "primary")
    output["hypotheses"]["V3"] = {
        "verdict": primary_v3["verdict"], "primary": True,
        "minimum_pooled_improvement": v3_cfg["minimum_pooled_improvement"],
        "comparisons": v3_results,
        "claim_boundary": "protocol extension is reported but cannot overturn the primary verdict",
    }

    v4_cfg = config["v4"]["primary"]
    strict_protocol = config["strict_removal_protocol"]
    strict_payload = load_strict_payload(
        v4_cfg["strict_path"], semantics, strict_protocol, allow_missing,
    )
    if strict_payload is None:
        output["hypotheses"]["V4"] = {"verdict": "pending", "primary": True}
    else:
        adaptive = strict_cohort(
            strict_payload, v4_cfg["adaptive_label"], seeds, episodes,
        )
        clean = strict_cohort(
            strict_payload, v4_cfg["clean_label"], seeds, episodes,
        )
        effect = strict_comparison(
            strict_payload, v4_cfg["adaptive_label"], v4_cfg["clean_label"],
            v4_cfg["branch"],
        )
        passed = effect["safe_paired_bootstrap_95"][0] > 0
        output["hypotheses"]["V4"] = {
            "verdict": verdict(passed), "primary": True,
            "adaptive_method": adaptive["method"],
            "clean_method": clean["method"],
            "branch": v4_cfg["branch"],
            "paired_effect": effect,
            "protocol": strict_protocol,
        }

    v5_cfg = config["v5"]
    strict_payload = load_strict_payload(
        v5_cfg["strict_path"], semantics, strict_protocol, allow_missing,
    )
    if strict_payload is None:
        output["hypotheses"]["V5"] = {"verdict": "pending", "primary": True}
    else:
        reference = strict_cohort(
            strict_payload, v5_cfg["reference_label"], seeds, episodes,
        )
        result = strict_cohort(
            strict_payload, v5_cfg["primary_label"], seeds, episodes,
        )
        thresholds = {
            "raw_success_rate": reference["success_rate"],
            "safe_success_rate": reference["safe_success_rate"],
            "constraint_violation_rate": reference["constraint_violation_rate"],
        }
        checks = {
            "raw": float(result["success_rate"]) >= thresholds["raw_success_rate"],
            "safe": float(result["safe_success_rate"]) >= thresholds["safe_success_rate"],
            "violation": float(result["constraint_violation_rate"])
            <= thresholds["constraint_violation_rate"],
        }
        candidate = {
            "label": "primary", "method": result["method"],
            "verdict": verdict(all(checks.values())), "checks": checks,
            "raw_success_rate": result["success_rate"],
            "safe_success_rate": result["safe_success_rate"],
            "constraint_violation_rate": result["constraint_violation_rate"],
            "paired_against_state": strict_comparison(
                strict_payload, v5_cfg["primary_label"], v5_cfg["reference_label"],
            ),
        }
        output["hypotheses"]["V5"] = {
            "verdict": candidate["verdict"], "primary": True,
            "state_reference": thresholds, "candidates": [candidate],
            "protocol": strict_protocol,
            "claim_boundary": "matched strict physical-removal endpoint; no target-only episodes",
        }
    output["all_primary_confirmed"] = all(
        item["verdict"] == "confirmed" for item in output["hypotheses"].values()
    )
    output["any_primary_pending"] = any(
        item["verdict"] == "pending" for item in output["hypotheses"].values()
    )
    return output


def markdown(payload):
    lines = [
        "# V3 visual-recovery hypothesis verdicts", "",
        "| Hypothesis | Primary verdict | Evidence boundary |", "|---|---:|---|",
    ]
    boundaries = {
        "V1": "Direct RGB PPO at the declared 70% gate; DAgger fallback reported separately",
        "V2": "Asymmetric versus symmetric direct RGB PPO; paired 95% interval",
        "V3": "Temporal versus matched non-temporal direct RGB PPO; extension shown separately",
        "V4": "Adaptive versus clean training when the first goal is physically removed",
        "V5": "Adaptive visual policy versus state policy on matched physical removals",
    }
    for name in ("V1", "V2", "V3", "V4", "V5"):
        lines.append(
            f"| {name} | {payload['hypotheses'][name]['verdict']} | {boundaries[name]} |"
        )
    lines.extend(["", "Generated from frozen aggregate JSON; training metrics are excluded.", ""])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="configs/visual_recovery_hypothesis_validation_v1.json"
    )
    parser.add_argument("--output", default="results/final_visual_comparison")
    parser.add_argument("--allow-missing", action="store_true")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    result = validate(config, args.allow_missing)
    root = Path(args.output)
    atomic_text(json.dumps(result, indent=2, sort_keys=True) + "\n", root / "hypotheses.json")
    atomic_text(markdown(result), root / "hypotheses.md")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

"""Cluster-ready, resumable execution for large benchmark experiments.

The older evaluation harness is intentionally small and in-process.  This
module adds the missing experiment contract around it: a versioned manifest,
deterministic content-addressed cases, stable sharding, one atomic artifact per
case/policy, strict validation, resumption, and paired aggregation.  It uses
only the standard library plus NumPy and keeps simulator imports lazy so
manifests and result sets can be validated on login/analysis nodes.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import csv
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import socket
import subprocess
import sys
import tempfile
import time
from typing import Callable, Iterable

import numpy as np

from atr.evaluation.harness import bootstrap_ci

SCHEMA_VERSION = 1
SUPPORTED_POLICIES = frozenset(
    {"static", "oracle_feasibility", "guarded_substitution", "unguarded_substitution"}
)
CORE_METRICS = (
    "goals_achieved",
    "total_steps",
    "wasted_steps",
    "constraint_violations",
    "navigation_replans",
    "navigation_safety_blocks",
    "navigation_failures",
)


@dataclass(frozen=True)
class BenchmarkCase:
    case_id: str
    env_id: str
    scene_variant: str | None
    intervention_kind: str
    onset_step_range: tuple[int, int]
    seed: int
    condition: str

    def payload(self) -> dict:
        data = asdict(self)
        data["onset_step_range"] = list(self.onset_step_range)
        return data


@dataclass(frozen=True)
class BenchmarkSpec:
    name: str
    policies: tuple[str, ...]
    metrics: tuple[str, ...]
    environments: tuple[dict, ...]
    seed_start: int
    seed_stop: int
    schema_version: int = SCHEMA_VERSION

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(_canonical_json(spec_to_dict(self)).encode()).hexdigest()[:16]


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _case_id(payload: dict) -> str:
    return hashlib.sha256(_canonical_json(payload).encode()).hexdigest()[:20]


def spec_to_dict(spec: BenchmarkSpec) -> dict:
    return {
        "schema_version": spec.schema_version,
        "name": spec.name,
        "policies": list(spec.policies),
        "metrics": list(spec.metrics),
        "seeds": {"start": spec.seed_start, "stop": spec.seed_stop},
        "environments": list(spec.environments),
    }


def load_spec(path: str | Path) -> BenchmarkSpec:
    raw = json.loads(Path(path).read_text())
    required = {"schema_version", "name", "policies", "metrics", "seeds", "environments"}
    unknown = set(raw) - required
    missing = required - set(raw)
    if missing or unknown:
        raise ValueError(f"manifest keys invalid: missing={sorted(missing)}, unknown={sorted(unknown)}")
    seeds = raw["seeds"]
    if set(seeds) != {"start", "stop"}:
        raise ValueError("seeds must contain exactly 'start' and 'stop'")
    spec = BenchmarkSpec(
        schema_version=int(raw["schema_version"]),
        name=str(raw["name"]),
        policies=tuple(raw["policies"]),
        metrics=tuple(raw["metrics"]),
        environments=tuple(raw["environments"]),
        seed_start=int(seeds["start"]),
        seed_stop=int(seeds["stop"]),
    )
    validate_spec(spec)
    return spec


def validate_spec(spec: BenchmarkSpec) -> None:
    if spec.schema_version != SCHEMA_VERSION:
        raise ValueError(f"unsupported schema_version={spec.schema_version}; expected {SCHEMA_VERSION}")
    if not spec.name.strip():
        raise ValueError("benchmark name cannot be empty")
    if spec.seed_start < 0 or spec.seed_stop <= spec.seed_start:
        raise ValueError("seeds must satisfy 0 <= start < stop")
    if not spec.policies or len(set(spec.policies)) != len(spec.policies):
        raise ValueError("policies must be non-empty and unique")
    unsupported = set(spec.policies) - SUPPORTED_POLICIES
    if unsupported:
        raise ValueError(f"unsupported policies: {sorted(unsupported)}")
    if not spec.metrics or len(set(spec.metrics)) != len(spec.metrics):
        raise ValueError("metrics must be non-empty and unique")
    if not spec.environments:
        raise ValueError("at least one environment matrix is required")
    for env in spec.environments:
        required = {"env_id", "scene_variants", "conditions"}
        if set(env) != required:
            raise ValueError(
                f"environment entry must contain exactly {sorted(required)}; got {sorted(env)}"
            )
        if not env["env_id"] or not env["scene_variants"] or not env["conditions"]:
            raise ValueError("environment id, scene_variants, and conditions cannot be empty")
        labels: set[str] = set()
        for condition in env["conditions"]:
            if set(condition) != {"label", "intervention_kind", "onset_step_range"}:
                raise ValueError("condition requires label, intervention_kind, onset_step_range")
            label_value = condition["label"]
            if not label_value or label_value in labels:
                raise ValueError(f"condition labels must be non-empty and unique: {label_value!r}")
            labels.add(label_value)
            onset = condition["onset_step_range"]
            if len(onset) != 2 or int(onset[0]) < 0 or int(onset[1]) <= int(onset[0]):
                raise ValueError(f"invalid onset_step_range: {onset!r}")


def expand_cases(spec: BenchmarkSpec) -> tuple[BenchmarkCase, ...]:
    validate_spec(spec)
    cases: list[BenchmarkCase] = []
    for env in spec.environments:
        for scene_variant in env["scene_variants"]:
            for condition in env["conditions"]:
                for seed in range(spec.seed_start, spec.seed_stop):
                    identity = {
                        "env_id": env["env_id"],
                        "scene_variant": scene_variant,
                        "intervention_kind": condition["intervention_kind"],
                        "onset_step_range": [int(x) for x in condition["onset_step_range"]],
                        "seed": seed,
                        "condition": condition["label"],
                    }
                    cases.append(
                        BenchmarkCase(
                            case_id=_case_id(identity),
                            env_id=identity["env_id"],
                            scene_variant=scene_variant,
                            intervention_kind=identity["intervention_kind"],
                            onset_step_range=tuple(identity["onset_step_range"]),
                            seed=seed,
                            condition=identity["condition"],
                        )
                    )
    ids = [case.case_id for case in cases]
    if len(ids) != len(set(ids)):
        raise ValueError("manifest expands to duplicate cases")
    return tuple(cases)


def pilot_spec(spec: BenchmarkSpec) -> BenchmarkSpec:
    """One seed in every matrix cell, with a distinct content fingerprint."""
    return BenchmarkSpec(
        name=f"{spec.name}_pilot",
        policies=spec.policies,
        metrics=spec.metrics,
        environments=spec.environments,
        seed_start=spec.seed_start,
        seed_stop=spec.seed_start + 1,
        schema_version=spec.schema_version,
    )


def shard_cases(
    cases: Iterable[BenchmarkCase], shard_index: int, shard_count: int,
) -> tuple[BenchmarkCase, ...]:
    if shard_count <= 0 or not 0 <= shard_index < shard_count:
        raise ValueError("shards require count > 0 and 0 <= index < count")
    return tuple(
        case for case in cases if int(case.case_id, 16) % shard_count == shard_index
    )


_ENV_ADAPTERS = {
    "TidyUp-v1": ("atr.envs.tidy_up_policies", "pd_ee_delta_pos"),
    "TidyUp-Humanoid-v1": ("atr.envs.tidy_up_humanoid_policies", "pd_joint_pos"),
    "TidyUp-ReplicaCAD-v1": ("atr.envs.tidy_up_replicacad_policies", "pd_ee_delta_pos"),
    "TidyUp-ReplicaCAD-Humanoid-v1": (
        "atr.envs.tidy_up_replicacad_humanoid_policies",
        "pd_joint_pos",
    ),
}


def _policy(module, name: str):
    if name == "static":
        return module.static_policy
    if name == "oracle_feasibility":
        return module.feasibility_aware_policy
    if name == "guarded_substitution":
        return lambda env: module.naive_substitution_policy(env, use_intent_guard=True)
    if name == "unguarded_substitution":
        return lambda env: module.naive_substitution_policy(env, use_intent_guard=False)
    raise ValueError(f"unsupported policy: {name}")


def _git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True,
            cwd=Path(__file__).resolve().parents[3],
        ).stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None


def _runtime_metadata() -> dict:
    packages = {}
    for distribution in ("mani-skill", "numpy", "gymnasium", "sapien"):
        try:
            packages[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            packages[distribution] = None
    return {
        "hostname": socket.gethostname(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": packages,
    }


def execute_case(case: BenchmarkCase, policy_name: str) -> dict:
    """Run one fresh simulator episode. Imports stay lazy for analysis nodes."""
    if case.env_id not in _ENV_ADAPTERS:
        raise ValueError(f"no environment adapter for {case.env_id!r}")
    import gymnasium as gym
    import task_schema_draft  # noqa: F401  (registers all project environments)

    module_name, control_mode = _ENV_ADAPTERS[case.env_id]
    policy_fn = _policy(importlib.import_module(module_name), policy_name)
    kwargs = {
        "num_envs": 1,
        "obs_mode": "state",
        "render_mode": None,
        "sim_backend": "physx_cpu",
        "control_mode": control_mode,
        "intervention_kind": case.intervention_kind,
        "onset_step_range": case.onset_step_range,
    }
    if case.scene_variant is not None:
        kwargs["scene_variant"] = case.scene_variant
    env = gym.make(case.env_id, **kwargs)
    try:
        env.reset(seed=case.seed)
        result = policy_fn(env)
        oracle_exists = dict(env.unwrapped._exists)
    finally:
        env.close()
    return {"outcome": _jsonable(result), "oracle_exists_final": _jsonable(oracle_exists)}


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _metric_values(outcome: dict) -> dict[str, float]:
    violations = sum(
        bool(value) for key, value in outcome.items() if key.endswith("_violated")
    )
    values = {
        "goals_achieved": float(outcome["goals_achieved"]),
        "total_steps": float(outcome["total_steps"]),
        "wasted_steps": float(outcome["wasted_steps"]),
        "constraint_violations": float(violations),
        "navigation_replans": float(outcome.get("navigation_replans", 0)),
        "navigation_safety_blocks": float(outcome.get("navigation_safety_blocks", 0)),
        "navigation_failures": float(outcome.get("navigation_failures", 0)),
    }
    if not all(np.isfinite(value) for value in values.values()):
        raise ValueError("outcome contains non-finite metrics")
    return values


def _record_path(records_dir: Path, case: BenchmarkCase, policy_name: str) -> Path:
    return records_dir / f"{case.case_id}__{policy_name}.json"


def _valid_completed_record(path: Path, case: BenchmarkCase, policy_name: str) -> bool:
    if not path.exists():
        return False
    try:
        record = json.loads(path.read_text())
        return (
            record["schema_version"] == SCHEMA_VERSION
            and record["status"] == "completed"
            and record["case"]["case_id"] == case.case_id
            and record["policy"] == policy_name
            and set(CORE_METRICS).issubset(record["metrics"])
        )
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        return False


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


Executor = Callable[[BenchmarkCase, str], dict]


def run_shard(
    spec: BenchmarkSpec,
    output_root: str | Path,
    *,
    shard_index: int = 0,
    shard_count: int = 1,
    executor: Executor = execute_case,
    fail_fast: bool = False,
) -> dict:
    """Run/resume one stable shard and return a machine-readable summary."""
    cases = shard_cases(expand_cases(spec), shard_index, shard_count)
    run_dir = Path(output_root) / f"{spec.name}__{spec.fingerprint}"
    records_dir = run_dir / "records"
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = run_dir / "manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != spec_to_dict(spec):
        raise ValueError(f"output run directory contains a different manifest: {run_dir}")
    _atomic_json(manifest_path, spec_to_dict(spec))

    counts = {"completed": 0, "resumed": 0, "failed": 0}
    commit = _git_commit()
    runtime = _runtime_metadata()
    for case in cases:
        for policy_name in spec.policies:
            path = _record_path(records_dir, case, policy_name)
            if _valid_completed_record(path, case, policy_name):
                counts["resumed"] += 1
                continue
            started = datetime.now(timezone.utc)
            started_clock = time.monotonic()
            try:
                execution = executor(case, policy_name)
                outcome = execution["outcome"]
                record = {
                    "schema_version": SCHEMA_VERSION,
                    "benchmark_name": spec.name,
                    "benchmark_fingerprint": spec.fingerprint,
                    "git_commit": commit,
                    "runtime": runtime,
                    "status": "completed",
                    "case": case.payload(),
                    "policy": policy_name,
                    "metrics": _metric_values(outcome),
                    "outcome": outcome,
                    "oracle_exists_final": execution.get("oracle_exists_final", {}),
                    "started_at": started.isoformat(),
                    "duration_seconds": time.monotonic() - started_clock,
                }
                counts["completed"] += 1
            except Exception as error:
                record = {
                    "schema_version": SCHEMA_VERSION,
                    "benchmark_name": spec.name,
                    "benchmark_fingerprint": spec.fingerprint,
                    "git_commit": commit,
                    "runtime": runtime,
                    "status": "failed",
                    "case": case.payload(),
                    "policy": policy_name,
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "started_at": started.isoformat(),
                    "duration_seconds": time.monotonic() - started_clock,
                }
                counts["failed"] += 1
            _atomic_json(path, record)
            if record["status"] == "failed" and fail_fast:
                raise RuntimeError(f"benchmark case failed: {path}: {record['error']}")

    return {
        "benchmark_name": spec.name,
        "benchmark_fingerprint": spec.fingerprint,
        "shard_index": shard_index,
        "shard_count": shard_count,
        "cases_in_shard": len(cases),
        "policy_runs_in_shard": len(cases) * len(spec.policies),
        **counts,
        "run_dir": str(run_dir),
    }


def load_completed_records(run_dir: str | Path) -> list[dict]:
    records = []
    for path in sorted((Path(run_dir) / "records").glob("*.json")):
        record = json.loads(path.read_text())
        if record.get("status") == "completed":
            records.append(record)
    return records


def validate_result_completeness(spec: BenchmarkSpec, records: list[dict]) -> None:
    """Require exactly one completed record for every manifest case/policy."""
    expected = {
        (case.case_id, policy)
        for case in expand_cases(spec)
        for policy in spec.policies
    }
    observed_list = [
        (record["case"]["case_id"], record["policy"])
        for record in records
    ]
    observed = set(observed_list)
    duplicates = len(observed_list) - len(observed)
    missing = sorted(expected - observed)
    extra = sorted(observed - expected)
    if duplicates or missing or extra:
        raise ValueError(
            "incomplete result set: "
            f"missing={missing[:5]}, extra={extra[:5]}, duplicates={duplicates}"
        )


def _aggregate_paired_group(
    records: list[dict], metrics: tuple[str, ...], reference_policy: str,
    n_resamples: int, ci: float,
) -> dict:
    """Aggregate one already-selected stratum with strict policy pairing."""
    by_policy: dict[str, dict[str, dict]] = {}
    for record in records:
        by_policy.setdefault(record["policy"], {})[record["case"]["case_id"]] = record
    if reference_policy not in by_policy:
        raise ValueError(f"reference policy {reference_policy!r} is absent")
    reference_ids = set(by_policy[reference_policy])
    report = {"reference_policy": reference_policy, "policies": {}}
    for policy_name, policy_records in sorted(by_policy.items()):
        case_ids = set(policy_records)
        if case_ids != reference_ids:
            missing = sorted(reference_ids - case_ids)
            extra = sorted(case_ids - reference_ids)
            raise ValueError(
                f"unpaired policy {policy_name!r}: missing={missing[:5]}, extra={extra[:5]}"
            )
        policy_report = {"n": len(case_ids), "metrics": {}, "paired_delta_vs_reference": {}}
        ordered_ids = sorted(case_ids)
        for metric in metrics:
            values = [float(policy_records[case_id]["metrics"][metric]) for case_id in ordered_ids]
            policy_report["metrics"][metric] = bootstrap_ci(
                values, n_resamples=n_resamples, ci=ci
            )
            deltas = [
                float(policy_records[case_id]["metrics"][metric])
                - float(by_policy[reference_policy][case_id]["metrics"][metric])
                for case_id in ordered_ids
            ]
            policy_report["paired_delta_vs_reference"][metric] = bootstrap_ci(
                deltas, n_resamples=n_resamples, ci=ci
            )
        report["policies"][policy_name] = policy_report
    return report


def aggregate_records(
    records: list[dict],
    *,
    metrics: tuple[str, ...],
    reference_policy: str = "oracle_feasibility",
    n_resamples: int = 2000,
    ci: float = 0.95,
) -> dict:
    """Aggregate overall and stratified CIs, rejecting incomplete pairing."""
    if not records:
        raise ValueError("no completed records to aggregate")
    report = _aggregate_paired_group(records, metrics, reference_policy, n_resamples, ci)
    strata: dict[str, list[dict]] = {}
    for record in records:
        case = record["case"]
        key = _canonical_json({
            "env_id": case["env_id"],
            "scene_variant": case["scene_variant"],
            "condition": case["condition"],
        })
        strata.setdefault(key, []).append(record)
    report["strata"] = {
        key: _aggregate_paired_group(group, metrics, reference_policy, n_resamples, ci)
        for key, group in sorted(strata.items())
    }
    return report


def write_summary_table_csv(report: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["policy", "metric", "n", "mean", "ci_low", "ci_high", "paired_delta_mean",
             "paired_delta_ci_low", "paired_delta_ci_high"]
        )
        for policy_name, policy in sorted(report["policies"].items()):
            for metric, interval in sorted(policy["metrics"].items()):
                delta = policy["paired_delta_vs_reference"][metric]
                writer.writerow([policy_name, metric, policy["n"], *interval, *delta])

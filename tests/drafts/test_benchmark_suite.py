"""Focused tests for the cluster-ready benchmark contract."""

import json

import pytest

from atr.evaluation.benchmark_suite import (
    BenchmarkSpec,
    aggregate_records,
    expand_cases,
    load_completed_records,
    load_spec,
    pilot_spec,
    run_shard,
    shard_cases,
    validate_result_completeness,
    write_summary_table_csv,
)
from atr.evaluation.benchmark_suite import _metric_values


def _spec(seed_stop=4):
    return BenchmarkSpec(
        name="test_matrix",
        policies=("static", "oracle_feasibility"),
        metrics=("goals_achieved", "wasted_steps", "constraint_violations"),
        environments=({
            "env_id": "TidyUp-v1",
            "scene_variants": [None],
            "conditions": [
                {
                    "label": "nominal",
                    "intervention_kind": "none",
                    "onset_step_range": [2, 3],
                },
                {
                    "label": "destroyed",
                    "intervention_kind": "bowl_destroyed",
                    "onset_step_range": [10, 60],
                },
            ],
        },),
        seed_start=0,
        seed_stop=seed_stop,
    )


def _executor(case, policy):
    destroyed = case.condition == "destroyed"
    oracle = policy == "oracle_feasibility"
    return {
        "outcome": {
            "per_goal": {},
            "goals_achieved": 1 if destroyed else 2,
            "total_steps": 25 if oracle and destroyed else 50,
            "wasted_steps": 0 if oracle or not destroyed else 25,
            "navigation_replans": 0,
            "navigation_safety_blocks": 0,
            "navigation_failures": 0,
        },
        "oracle_exists_final": {"blue_bowl": not destroyed},
    }


def test_manifest_expansion_is_deterministic_unique_and_content_addressed():
    first = expand_cases(_spec())
    second = expand_cases(_spec())
    assert first == second
    assert len(first) == 8
    assert len({case.case_id for case in first}) == 8
    assert all(len(case.case_id) == 20 for case in first)


def test_stable_shards_form_an_exact_partition():
    cases = expand_cases(_spec(seed_stop=20))
    shards = [shard_cases(cases, index, 7) for index in range(7)]
    flattened = [case.case_id for shard in shards for case in shard]
    assert len(flattened) == len(cases)
    assert set(flattened) == {case.case_id for case in cases}
    assert len(flattened) == len(set(flattened))


def test_pilot_keeps_every_matrix_cell_but_only_one_seed():
    full = _spec(seed_stop=20)
    pilot = pilot_spec(full)
    assert pilot.name.endswith("_pilot")
    assert pilot.fingerprint != full.fingerprint
    assert len(expand_cases(pilot)) == 2
    assert {case.condition for case in expand_cases(pilot)} == {"nominal", "destroyed"}


def test_invalid_manifest_fails_before_any_simulator_work(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({
        "schema_version": 1,
        "name": "bad",
        "policies": ["imaginary_policy"],
        "metrics": ["goals_achieved"],
        "seeds": {"start": 0, "stop": 1},
        "environments": [{
            "env_id": "TidyUp-v1",
            "scene_variants": [None],
            "conditions": [{
                "label": "bad_range",
                "intervention_kind": "none",
                "onset_step_range": [3, 3],
            }],
        }],
    }))
    with pytest.raises(ValueError, match="unsupported policies"):
        load_spec(path)


def test_runner_writes_atomic_records_and_resumes_completed_work(tmp_path):
    calls = []

    def executor(case, policy):
        calls.append((case.case_id, policy))
        return _executor(case, policy)

    first = run_shard(_spec(seed_stop=2), tmp_path, executor=executor)
    assert first["completed"] == 8
    assert first["failed"] == 0
    assert len(calls) == 8

    second = run_shard(_spec(seed_stop=2), tmp_path, executor=executor)
    assert second["completed"] == 0
    assert second["resumed"] == 8
    assert len(calls) == 8
    records = load_completed_records(first["run_dir"])
    assert len(records) == 8
    assert all(record["git_commit"] is None or len(record["git_commit"]) == 40 for record in records)
    assert all(record["runtime"]["python"] for record in records)
    assert all("mani-skill" in record["runtime"]["packages"] for record in records)


def test_failed_record_is_visible_and_retried_on_resume(tmp_path):
    attempts = 0

    def flaky(case, policy):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("synthetic worker failure")
        return _executor(case, policy)

    first = run_shard(_spec(seed_stop=1), tmp_path, executor=flaky)
    assert first["failed"] == 1
    assert first["completed"] == 3
    second = run_shard(_spec(seed_stop=1), tmp_path, executor=flaky)
    assert second["completed"] == 1
    assert second["resumed"] == 3
    assert second["failed"] == 0


def test_aggregation_requires_pairing_and_exports_summary_csv(tmp_path):
    summary = run_shard(_spec(seed_stop=4), tmp_path, executor=_executor)
    records = load_completed_records(summary["run_dir"])
    report = aggregate_records(
        records,
        metrics=("goals_achieved", "wasted_steps", "constraint_violations"),
        n_resamples=200,
    )
    assert report["policies"]["static"]["n"] == 8
    assert len(report["strata"]) == 2
    # Four destroyed cases differ by +25 waste relative to oracle; four
    # nominal cases tie, so the paired mean is exactly 12.5.
    assert report["policies"]["static"]["paired_delta_vs_reference"]["wasted_steps"][0] == 12.5
    path = tmp_path / "summary_table.csv"
    write_summary_table_csv(report, path)
    assert path.read_text().splitlines()[0].startswith("policy,metric,n,mean")
    assert "static,wasted_steps" in path.read_text()

    unpaired = [record for record in records if not (
        record["policy"] == "static" and record["case"]["seed"] == 0
    )]
    with pytest.raises(ValueError, match="unpaired policy"):
        aggregate_records(unpaired, metrics=("wasted_steps",), n_resamples=20)


def test_completeness_check_detects_globally_missing_case(tmp_path):
    spec = _spec(seed_stop=2)
    summary = run_shard(spec, tmp_path, executor=_executor)
    records = load_completed_records(summary["run_dir"])
    validate_result_completeness(spec, records)
    # Removing both policies for one case remains internally paired, so only
    # manifest-aware completeness validation can catch it.
    missing_case = records[0]["case"]["case_id"]
    incomplete = [r for r in records if r["case"]["case_id"] != missing_case]
    with pytest.raises(ValueError, match="incomplete result set"):
        validate_result_completeness(spec, incomplete)


def test_constraint_metric_prefers_uniform_oracle_evaluation():
    outcome = {
        "goals_achieved": 1,
        "total_steps": 2,
        "wasted_steps": 0,
        # A policy-specific flag can be absent or even wrong; all real
        # policies must be scored from the same environment-oracle map.
        "dont_move_glass_violated": False,
        "oracle_constraint_violations": {
            "dont_move_glass": True,
            "keep_medicine_upright": False,
        },
    }
    assert _metric_values(outcome)["constraint_violations"] == 1.0

"""Tests for atr.evaluation.tracking (D-057) -- experiment tracking on top
of the already-real harness (D-042) and log interface (D-056). Real live
episodes (small: 2 seeds, 2 policies) rather than mocked, same as every
other integration test in this project -- the thing actually worth
verifying is that a real run gets persisted and is queryable afterward,
not that a mock got called correctly.
"""

import json

import gymnasium as gym
import pytest

pytest.importorskip("mani_skill")

import task_schema_draft  # noqa: E402, F401
from atr.envs.tidy_up_policies import feasibility_aware_policy, static_policy  # noqa: E402
from atr.evaluation.tracking import list_runs, track_comparison  # noqa: E402
from atr.language.goal_graph import canonical_example  # noqa: E402


def _make_env():
    return gym.make(
        "TidyUp-v1", num_envs=1, obs_mode="state", render_mode=None,
        sim_backend="physx_cpu", control_mode="pd_ee_delta_pos",
        intervention_kind="bowl_destroyed", onset_step_range=(2, 3),
    )


class TestTrackComparison:
    def test_writes_a_summary_with_a_real_report_and_metadata(self, tmp_path):
        summary = track_comparison(
            "static_vs_feasibility_aware",
            _make_env,
            {"static": static_policy, "feasibility_aware": feasibility_aware_policy},
            seeds=[0, 1],
            graph=canonical_example(),
            n_resamples=200,
            runs_dir=tmp_path,
        )

        assert summary["run_name"] == "static_vs_feasibility_aware"
        assert summary["run_id"].endswith("_static_vs_feasibility_aware")
        assert summary["seeds"] == [0, 1]
        assert summary["policy_names"] == ["feasibility_aware", "static"]
        assert summary["instruction_text"] == canonical_example().instruction_text
        # git_commit is best-effort (None outside a git checkout) -- just
        # check it's a string when present, not a specific value.
        assert summary["git_commit"] is None or isinstance(summary["git_commit"], str)
        assert "static" in summary["report"]
        assert "wasted_steps" in summary["report"]["static"]

    def test_summary_json_on_disk_matches_the_return_value(self, tmp_path):
        summary = track_comparison(
            "disk_check", _make_env, {"static": static_policy}, seeds=[0],
            graph=canonical_example(), n_resamples=200, runs_dir=tmp_path,
        )
        run_dir = tmp_path / summary["run_id"]
        on_disk = json.loads((run_dir / "summary.json").read_text())
        # JSON has no tuple type -- report's (mean, lo, hi) tuples come
        # back as lists, so compare through the same round-trip rather
        # than a direct == against the in-memory summary.
        assert on_disk == json.loads(json.dumps(summary))

    def test_also_writes_per_policy_episode_logs(self, tmp_path):
        summary = track_comparison(
            "logs_check", _make_env, {"static": static_policy}, seeds=[0, 1],
            graph=canonical_example(), n_resamples=200, runs_dir=tmp_path,
        )
        run_dir = tmp_path / summary["run_id"]
        log_lines = (run_dir / "static.jsonl").read_text().strip().splitlines()
        assert len(log_lines) == 2  # one per seed
        first_record = json.loads(log_lines[0])
        assert first_record["seed"] == 0
        assert "per_goal" in first_record


class TestListRuns:
    def test_empty_when_nothing_tracked_yet(self, tmp_path):
        assert list_runs(tmp_path / "does_not_exist") == []

    def test_finds_tracked_runs_oldest_first(self, tmp_path):
        first = track_comparison(
            "run_a", _make_env, {"static": static_policy}, seeds=[0],
            graph=canonical_example(), n_resamples=200, runs_dir=tmp_path,
        )
        second = track_comparison(
            "run_b", _make_env, {"static": static_policy}, seeds=[0],
            graph=canonical_example(), n_resamples=200, runs_dir=tmp_path,
        )

        runs = list_runs(tmp_path)
        assert [r["run_id"] for r in runs] == [first["run_id"], second["run_id"]]

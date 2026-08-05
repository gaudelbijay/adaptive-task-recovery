"""Tests for atr.evaluation.logging (D-056) -- the log interface
docs/03-system-architecture.md's data-flow step 6 described but nothing
ever built. Pure-function, no simulator needed -- exercises
build_episode_log() against synthetic result dicts shaped exactly like
the real ones baselines._summarize()/run_end_to_end_episode(s) produce,
not against a live env (that's test_evaluation_harness.py's job).
"""

import numpy as np
import pytest

from atr.evaluation.logging import append_episode_log, build_episode_log, read_episode_logs
from atr.language.goal_graph import Constraint, Goal, GoalGraph

_GRAPH = GoalGraph(
    instruction_text="Put the mug and the bowl on the tray, do not move the glass.",
    goals=(
        Goal(id="place_mug", predicate="on_tray", target_object="mug"),
        Goal(id="place_bowl", predicate="on_tray", target_object="bowl"),
    ),
    constraints=(
        Constraint(id="dont_move_glass", kind="never_move", target_object="glass"),
    ),
)


class TestBuildEpisodeLog:
    def test_attaches_oracle_label_per_goal(self):
        result = {
            "per_goal": {
                "place_mug": {"achieved": True, "steps_used": 10, "skipped": False},
                "place_bowl": {"achieved": False, "steps_used": 0, "skipped": True},
            },
            "goals_achieved": 1, "total_steps": 10, "wasted_steps": 0,
        }
        oracle_exists = {"mug": True, "bowl": False, "glass": True}

        log = build_episode_log(result, _GRAPH, oracle_exists, seed=3, policy_name="feasibility_aware")

        assert log["seed"] == 3
        assert log["policy_name"] == "feasibility_aware"
        assert log["instruction_text"] == _GRAPH.instruction_text
        assert log["per_goal"]["place_mug"]["target_object"] == "mug"
        assert log["per_goal"]["place_mug"]["oracle_feasible"] is True
        assert log["per_goal"]["place_bowl"]["target_object"] == "bowl"
        assert log["per_goal"]["place_bowl"]["oracle_feasible"] is False
        # the original per_goal outcome fields survive untouched
        assert log["per_goal"]["place_mug"]["achieved"] is True
        assert log["per_goal"]["place_bowl"]["skipped"] is True
        assert log["goals_achieved"] == 1
        assert log["wasted_steps"] == 0

    def test_normalizes_any_violated_suffix_key_into_violations(self):
        # naive_substitution_policy's real shape: a dynamic dont_move_<object>_violated
        # key, not something callers should have to know the exact name of.
        result = {
            "per_goal": {"place_mug": {"achieved": False, "steps_used": 5, "skipped": False}},
            "goals_achieved": 0, "total_steps": 5, "wasted_steps": 5,
            "dont_move_glass_violated": True,
            "substitution_attempted": True,
        }
        log = build_episode_log(result, _GRAPH, {"mug": False, "glass": True})
        assert log["violations"] == {"dont_move_glass_violated": True}
        # non-violation extra keys (substitution_attempted) aren't lost either --
        # they're just not classified as a violation, since they don't end in _violated
        assert "substitution_attempted" not in log["violations"]

    def test_numpy_scalar_outcomes_are_converted_to_native_python(self):
        # real bug this guards against: goal_achieved() returns np.bool_,
        # which json.dumps rejects outright -- found investigating D-055.
        result = {
            "per_goal": {
                "place_mug": {"achieved": np.True_, "steps_used": np.int64(7), "skipped": False},
            },
            "goals_achieved": np.int64(1), "total_steps": np.int64(7), "wasted_steps": 0,
        }
        log = build_episode_log(result, _GRAPH, {"mug": True})
        assert isinstance(log["per_goal"]["place_mug"]["achieved"], bool)
        assert isinstance(log["per_goal"]["place_mug"]["steps_used"], int)
        assert isinstance(log["goals_achieved"], int)
        import json
        json.dumps(log)  # must not raise

    def test_unrecognized_goal_id_gets_no_target_object_or_oracle_label(self):
        # naive_substitution_policy's synthetic "substitute_for_X" goals
        # aren't in the real GoalGraph -- shouldn't crash, just log unknowns.
        result = {
            "per_goal": {"substitute_for_place_bowl": {"achieved": False, "steps_used": 3, "skipped": False}},
            "goals_achieved": 0, "total_steps": 3, "wasted_steps": 3,
        }
        log = build_episode_log(result, _GRAPH, {"mug": True})
        entry = log["per_goal"]["substitute_for_place_bowl"]
        assert entry["target_object"] is None
        assert entry["oracle_feasible"] is None


class TestJsonlRoundTrip:
    def test_append_then_read_recovers_every_record_in_order(self, tmp_path):
        path = tmp_path / "episodes.jsonl"
        records = [
            build_episode_log(
                {
                    "per_goal": {"place_mug": {"achieved": True, "steps_used": 1, "skipped": False}},
                    "goals_achieved": 1, "total_steps": 1, "wasted_steps": 0,
                },
                _GRAPH, {"mug": True}, seed=seed,
            )
            for seed in range(3)
        ]
        for record in records:
            append_episode_log(path, record)

        recovered = read_episode_logs(path)
        assert [r["seed"] for r in recovered] == [0, 1, 2]
        assert recovered == records

    def test_survives_a_path_given_as_a_plain_string(self, tmp_path):
        path = str(tmp_path / "episodes.jsonl")
        record = build_episode_log(
            {
                "per_goal": {"place_mug": {"achieved": True, "steps_used": 1, "skipped": False}},
                "goals_achieved": 1, "total_steps": 1, "wasted_steps": 0,
            },
            _GRAPH, {"mug": True}, seed=0,
        )
        append_episode_log(path, record)
        assert read_episode_logs(path) == [record]

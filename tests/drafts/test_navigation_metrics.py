"""D-094: aggregate and persist D-093 navigation-adaptation metadata."""

from atr.evaluation.logging import build_episode_log
from atr.language.goal_graph import canonical_example
from atr.policies.baselines import _summarize


def _result():
    return _summarize({
        "place_mug": {
            "achieved": True,
            "steps_used": 8,
            "skipped": False,
            "navigation_replanned": True,
            "predicted_affected_objects": ["glass"],
        },
        "place_bowl": {
            "achieved": False,
            "steps_used": 0,
            "skipped": True,
            "navigation_replanned": True,
            "navigation_safety_screened": True,
            "blocked_reason": "blocked: no safe route",
            "predicted_affected_objects": ["glass"],
        },
    })


def test_policy_summary_counts_replans_and_fail_closed_stops():
    result = _result()

    assert result["navigation_replans"] == 2
    assert result["navigation_safety_blocks"] == 1


def test_episode_log_preserves_navigation_aggregates_and_per_goal_evidence():
    log = build_episode_log(
        _result(),
        canonical_example(),
        {"mug": True, "bowl": True},
        seed=3,
        policy_name="constraint_aware",
    )

    assert log["navigation_replans"] == 2
    assert log["navigation_safety_blocks"] == 1
    assert log["per_goal"]["place_mug"]["predicted_affected_objects"] == ["glass"]


def test_embodiments_without_navigation_metadata_report_zero_adaptation():
    result = _summarize({
        "place_mug": {"achieved": True, "steps_used": 2, "skipped": False},
    })

    assert result["navigation_replans"] == 0
    assert result["navigation_safety_blocks"] == 0


def test_non_navigation_intent_guard_block_is_not_counted_as_navigation_block():
    result = _summarize({
        "place_bowl": {
            "achieved": False,
            "steps_used": 0,
            "skipped": True,
            "blocked_reason": "blocked: would violate dont_move_glass",
        },
    })

    assert result["navigation_safety_blocks"] == 0

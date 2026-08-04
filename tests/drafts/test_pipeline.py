"""Tests for atr.pipeline (promoted from end_to_end.py, D-050) -- stage 6 of docs/00-project-overview.md's
build-up order ("combine everything end-to-end"): language parsing, real
vision-based feasibility (not privileged state), and a learned policy, all
in one real episode.

Exactly two render-producing calls per episode (one per goal), matching
D-022's verified-safe budget -- this file runs exactly one episode per
test, never more, so it never risks the desync D-022 describes.
"""

import gymnasium as gym
import pytest

pytest.importorskip("mani_skill")
pytest.importorskip("open_clip")

import task_schema_draft  # noqa: E402, F401
from atr.pipeline import (  # noqa: E402
    run_end_to_end_episode,
    train_q_table_replicacad_humanoid,
)

# Trained once for the whole module -- ~2 minutes, privileged-state only
# (no rendering), see atr/pipeline.py's module docstring for why training
# doesn't need to be this slow or this rendering-heavy.
_Q_TABLE = None


def _q_table():
    global _Q_TABLE
    if _Q_TABLE is None:
        _Q_TABLE = train_q_table_replicacad_humanoid(n_episodes=30, seed=0)
    return _Q_TABLE


def _make_env(**kwargs):
    return gym.make(
        "TidyUp-ReplicaCAD-Humanoid-v1", num_envs=1, obs_mode="state",
        render_mode="rgb_array", sim_backend="physx_cpu", control_mode="pd_joint_pos", **kwargs,
    )


class TestFullPipelineMatchesOracle:
    """The actual claim of this stage: with nothing privileged in the live
    decision loop (perception comes from a rendered frame, the decision
    from a table trained on reward, not a hard-coded rule), the pipeline
    still gets the same answer oracle-feasibility policies do."""

    def test_destroyed_object_skipped_survivor_achieved(self):
        env = _make_env(intervention_kind="chef_can_destroyed", onset_step_range=(2, 3))
        try:
            env.reset(seed=0)
            result = run_end_to_end_episode(env, _q_table())
            oracle_exists = dict(env.unwrapped._exists)
        finally:
            env.close()

        assert oracle_exists["master_chef_can"] is False
        assert oracle_exists["potted_meat_can"] is True

        by_object = {
            "potted_meat_can": result["per_goal"]["place_potted_meat_can"],
            "master_chef_can": result["per_goal"]["place_master_chef_can"],
        }
        assert by_object["potted_meat_can"]["achieved"]
        assert by_object["potted_meat_can"]["perceived_feasible"] is True
        assert by_object["master_chef_can"]["skipped"] is True
        assert by_object["master_chef_can"]["perceived_feasible"] is False
        assert result["wasted_steps"] == 0
        assert result["goals_achieved"] == 1

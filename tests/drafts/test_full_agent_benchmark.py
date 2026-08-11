"""D-088: the project's own stated success criterion (docs/01 "Success
criteria") -- a multi-seed, bootstrap-CI comparison of the *full agent*
(real language parsing, real CLIP-perceived feasibility, a trained Q-table,
real arm motion) against a static baseline, with oracle-feasibility
performance as the headroom reference -- had never actually been run.
Running it for the first time surfaced a real, unplanned finding: CLIP's
crop-based feasibility judgment for `master_chef_can` in the kitchen_cabinet
scene, calibrated on clean reference frames (D-020/D-027), has a severe
false-negative gap once evaluated in this exact live-pipeline context --
*after* a real prior goal attempt has moved G1's arm into the calibrated
crop region. Visually confirmed (not just measured): the object is clearly
visible in the crop; the arm/hand now occupies much of it. Structurally the
same mechanism D-054 found for DINOv2 (a calibration done on frames unlike
what the live loop actually renders), in the opposite direction (a false
negative instead of a false positive), and never tested for CLIP in this
exact env/scene/post-attempt context before -- CLIP's D-020/D-027
validation predates any live decision loop.

Not fixed here, following D-054's own precedent (disclosed as a real,
informative negative result first; D-055 fixed it as a distinct follow-up
decision) -- CLIP is zero-shot, so "retrain on more representative
examples" doesn't directly apply the way it did for DINOv2's linear probe;
a fix would mean recalibrating the crop/prompt itself, a separate decision.

This IS the honest first run of docs/01's success-criteria benchmark: it
decomposes perception failure from policy failure exactly as docs/10 asks
("Decompose end-to-end failure into perception, feasibility, high-level
strategy...") -- the gap between `full_agent` and `oracle_feasibility` here
reflects a measured perception bottleneck, not a policy bug, and reporting
both together (not just `full_agent` alone) is what makes that legible.
"""

import gymnasium as gym
import pytest

pytest.importorskip("mani_skill")

import task_schema_draft  # noqa: E402, F401
from atr.envs.tidy_up_replicacad_humanoid_policies import _TRAY_SLOTS, attempt_goal  # noqa: E402
from atr.evaluation.full_agent_benchmark import (  # noqa: E402
    instruction_graph,
    run_full_agent_benchmark,
    serialize_q_table,
)
from atr.feasibility.oracle import goal_feasible  # noqa: E402
from atr.pipeline import train_q_table_replicacad_humanoid  # noqa: E402

_WIDE_ONSET_RANGE = (10, 60)


class TestSerializeQTableRoundTrips:
    def test_round_trips_through_the_same_deserializer_the_subprocess_uses(self):
        from atr.envs.run_full_agent_episode_subprocess import load_q_table
        import json
        import tempfile
        from pathlib import Path

        q_table = train_q_table_replicacad_humanoid(n_episodes=10, seed=0)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "q.json"
            path.write_text(json.dumps(serialize_q_table(q_table)))
            reloaded = load_q_table(str(path))
        assert reloaded.keys() == q_table.keys()
        for key in q_table:
            assert reloaded[key] == q_table[key]


class TestFullAgentBenchmarkRunsEndToEnd:
    """The actual success-criteria benchmark: static, oracle_feasibility,
    and the real full agent, paired across the same seeds, real bootstrap
    CIs -- docs/01's own definition of project success, run for the first
    time."""

    @pytest.fixture(scope="class")
    def q_table(self):
        return train_q_table_replicacad_humanoid()

    def test_returns_bootstrap_ci_for_every_policy_and_metric(self, q_table):
        result = run_full_agent_benchmark(
            seeds=list(range(5)), q_table=q_table, onset_step_range=_WIDE_ONSET_RANGE,
        )
        assert set(result.keys()) == {"static", "oracle_feasibility", "full_agent"}
        for policy_result in result.values():
            assert set(policy_result.keys()) == {"goals_achieved", "wasted_steps"}
            for mean, lo, hi in policy_result.values():
                assert lo <= mean <= hi

    def test_oracle_feasibility_outperforms_full_agent_here(self, q_table):
        """The real, measured, decomposed finding: not a policy bug -- the
        Q-table itself is reward-consistent (see the diagnostic scripts
        referenced in ai-notes/decisions.md's D-088 entry) -- but a real
        CLIP perception gap in this exact live-pipeline context. Reported
        honestly rather than only publishing `full_agent`'s number alone,
        exactly the decomposition docs/10 asks for."""
        result = run_full_agent_benchmark(
            seeds=list(range(5)), q_table=q_table, onset_step_range=_WIDE_ONSET_RANGE,
        )
        oracle_mean = result["oracle_feasibility"]["goals_achieved"][0]
        full_agent_mean = result["full_agent"]["goals_achieved"][0]
        assert oracle_mean > full_agent_mean


class TestClipFalseNegativeAfterALiveFirstAttempt:
    """The finding itself, isolated from the full benchmark: does CLIP's
    calibrated crop for `master_chef_can` (kitchen_cabinet) correctly judge
    it as present once G1 has completed a real attempt_goal() on the first
    goal, the exact state the live pipeline's second-goal decision actually
    renders? Measured directly against privileged oracle state, not
    asserted from the aggregate benchmark result alone."""

    def _make_env(self, onset_step_range, render_mode):
        return gym.make(
            "TidyUp-ReplicaCAD-Humanoid-v1", num_envs=1, obs_mode="state",
            render_mode=render_mode, sim_backend="physx_cpu", control_mode="pd_joint_pos",
            intervention_kind="chef_can_destroyed", onset_step_range=onset_step_range,
        )

    def test_clip_mismatches_oracle_on_most_genuinely_feasible_cases(self, tmp_path):
        """D-022's confirmed rendering-desync bug means a render-producing
        reset must not repeat more than ~2 times in one process -- an
        earlier version of this test rendered directly across 5 in-process
        resets and got a materially different, unreliable result (1/4
        mismatches instead of the real 7/7 measured via subprocess
        isolation). The oracle half never renders, so it stays in-process
        (safe across any number of seeds, same as every other
        privileged-state check in this project); the CLIP half must run
        one fresh subprocess per seed, `run_full_agent_episode_subprocess.py`
        (D-088), the same isolation `capture_episode_subprocess.py` (D-052)
        already established for exactly this bug."""
        import json
        import subprocess
        import sys

        from atr.evaluation.full_agent_benchmark import _SUBPROCESS_SCRIPT
        from atr.pipeline import train_q_table_replicacad_humanoid

        graph = instruction_graph()
        goal1, goal2 = graph.goals[0], graph.goals[1]

        q_table_path = tmp_path / "q_table.json"
        q_table_path.write_text(json.dumps(serialize_q_table(train_q_table_replicacad_humanoid())))

        mismatches = 0
        total_feasible = 0
        for seed in range(5):
            env = self._make_env(_WIDE_ONSET_RANGE, render_mode=None)
            try:
                env.reset(seed=seed)
                attempt_goal(env, goal1, _TRAY_SLOTS[0])
                oracle_feasible = bool(goal_feasible(goal2, env.unwrapped._world_state()))
            finally:
                env.close()
            if not oracle_feasible:
                continue
            total_feasible += 1

            out_path = tmp_path / f"episode_{seed}.json"
            subprocess.run(
                [
                    sys.executable, str(_SUBPROCESS_SCRIPT),
                    "--seed", str(seed), "--onset-step-min", str(_WIDE_ONSET_RANGE[0]),
                    "--onset-step-max", str(_WIDE_ONSET_RANGE[1]),
                    "--q-table-path", str(q_table_path), "--out", str(out_path),
                ],
                check=True,
            )
            episode = json.loads(out_path.read_text())
            clip_perceived = episode["per_goal"]["place_master_chef_can"]["perceived_feasible"]
            if clip_perceived != oracle_feasible:
                mismatches += 1

        # Real measured: 7/7 mismatches among genuinely feasible cases
        # across an 8-seed sample (D-088) -- CLIP said "absent" every time,
        # regardless of the true state. Loose bound here (majority, not
        # exact equality), same reasoning every other stochastic
        # real-world assertion in this project uses.
        assert total_feasible > 0
        assert mismatches / total_feasible > 0.5

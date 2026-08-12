"""D-088/D-089: the project's own stated success criterion (docs/01 "Success
criteria") -- a multi-seed, bootstrap-CI comparison of the *full agent*
(real language parsing, real CLIP-perceived feasibility, a trained Q-table,
real arm motion) against a static baseline, with oracle-feasibility
performance as the headroom reference -- had never actually been run.

Running it for the first time (D-088) surfaced a real, unplanned finding:
CLIP's crop-based feasibility judgment for `master_chef_can` in the
kitchen_cabinet scene, calibrated on clean reference frames (D-020/D-027),
had a severe false-negative gap once evaluated in this exact live-pipeline
context -- *after* a real prior goal attempt had moved G1's arm into the
calibrated crop region. CLIP said "absent" in all 8 episodes tested,
including all 7 that were genuinely feasible by oracle ground truth.
Visually confirmed, not just measured: the object was clearly visible in
the crop; the arm/hand occupied much of the same region, dominating the
crop's overall gist enough to flip CLIP's zero-shot margin. Structurally
the same mechanism D-054 found for DINOv2, in the opposite direction (a
false negative, not a false positive), never tested for CLIP in this exact
env/scene/post-attempt context before.

Fixed for real (D-089), following D-055's precedent for the analogous
DINOv2 gap: recalibrated `master_chef_can`'s kitchen_cabinet crop (prompt
left unchanged -- the fix is purely geometric) to a tighter region that
still reliably contains the object while excluding most of where G1's arm
ends up after a real completed attempt. Found by measuring several
candidate crops directly against saved present/absent frames (not
guessed), then validated against the original 8-seed sample: 0/8
mismatches (down from 7/8), and re-confirmed the pre-existing arm-at-rest
calibration test (`test_clip_feasibility.py`, D-020's original case) still
passes unchanged -- the fix generalizes across both visual states, not just
the one that was broken. Re-running the full benchmark with the fixed
calibration gives the actual success-criteria result: `full_agent` now
matches `oracle_feasibility` exactly on every metric across 15 seeds, and
both meaningfully beat `static` on `wasted_steps` while matching it on
`goals_achieved` -- the real, positive H2 confirmation this benchmark was
built to demonstrate.
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
    time, and (D-089) with a real, working CLIP calibration underneath it."""

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

    def test_full_agent_matches_oracle_feasibility_now(self, q_table):
        """The real, measured, positive result (D-089): with CLIP's
        post-attempt crop bug fixed, the real perceptual pipeline performs
        identically to the privileged-state headroom reference across
        every seed -- not merely "close," an exact match on both metrics
        in the real 15-seed run this test's 5-seed version mirrors at
        smaller scale (ai-notes/decisions.md D-089)."""
        result = run_full_agent_benchmark(
            seeds=list(range(5)), q_table=q_table, onset_step_range=_WIDE_ONSET_RANGE,
        )
        oracle_goals = result["oracle_feasibility"]["goals_achieved"][0]
        full_agent_goals = result["full_agent"]["goals_achieved"][0]
        oracle_wasted = result["oracle_feasibility"]["wasted_steps"][0]
        full_agent_wasted = result["full_agent"]["wasted_steps"][0]
        assert full_agent_goals == oracle_goals
        assert full_agent_wasted == oracle_wasted

    def test_feasibility_aware_policies_waste_fewer_steps_than_static(self, q_table):
        """H2's actual comparative claim, in the real perceptual pipeline
        for the first time: conditioning on feasibility saves wasted
        effort without sacrificing goal completion -- goals_achieved is
        identical to static (neither approach can complete a genuinely
        infeasible goal), but wasted_steps is measurably lower, since
        skipping a doomed goal costs nothing while a failed attempt costs
        real steps."""
        result = run_full_agent_benchmark(
            seeds=list(range(5)), q_table=q_table, onset_step_range=_WIDE_ONSET_RANGE,
        )
        assert result["full_agent"]["wasted_steps"][0] < result["static"]["wasted_steps"][0]
        assert result["full_agent"]["goals_achieved"][0] == result["static"]["goals_achieved"][0]


class TestClipCorrectlyJudgesAfterALiveFirstAttempt:
    """D-089: the recalibrated crop for `master_chef_can` (kitchen_cabinet)
    now correctly judges it as present once G1 has completed a real
    attempt_goal() on the first goal -- the exact state the live pipeline's
    second-goal decision actually renders. Measured directly against
    privileged oracle state, not asserted from the aggregate benchmark
    result alone. This locks in the fix for the exact case
    `test_clip_feasibility.py`'s pre-existing arm-at-rest tests don't
    cover."""

    def _make_env(self, onset_step_range, render_mode):
        return gym.make(
            "TidyUp-ReplicaCAD-Humanoid-v1", num_envs=1, obs_mode="state",
            render_mode=render_mode, sim_backend="physx_cpu", control_mode="pd_joint_pos",
            intervention_kind="chef_can_destroyed", onset_step_range=onset_step_range,
        )

    def test_clip_matches_oracle_after_a_real_first_attempt(self, tmp_path):
        """D-022's confirmed rendering-desync bug means a render-producing
        reset must not repeat more than ~2 times in one process -- an
        earlier version of this test rendered directly across 5 in-process
        resets and got a materially different, unreliable result. The
        oracle half never renders, so it stays in-process (safe across any
        number of seeds, same as every other privileged-state check in
        this project); the CLIP half must run one fresh subprocess per
        seed, `run_full_agent_episode_subprocess.py` (D-088), the same
        isolation `capture_episode_subprocess.py` (D-052) already
        established for exactly this bug."""
        import json
        import subprocess
        import sys

        from atr.evaluation.full_agent_benchmark import _SUBPROCESS_SCRIPT

        graph = instruction_graph()
        goal1, goal2 = graph.goals[0], graph.goals[1]

        q_table_path = tmp_path / "q_table.json"
        q_table_path.write_text(json.dumps(serialize_q_table(train_q_table_replicacad_humanoid())))

        mismatches = 0
        total_seeds = 0
        for seed in range(5):
            env = self._make_env(_WIDE_ONSET_RANGE, render_mode=None)
            try:
                env.reset(seed=seed)
                attempt_goal(env, goal1, _TRAY_SLOTS[0])
                oracle_feasible = bool(goal_feasible(goal2, env.unwrapped._world_state()))
            finally:
                env.close()
            total_seeds += 1

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

        # Real measured: 0/8 mismatches after the D-089 crop fix (down from
        # 7/8 before it). Asserting zero here, not a loose bound: this is
        # exactly the case the fix targeted and was validated against.
        assert total_seeds > 0
        assert mismatches == 0

"""D-088: the project's own stated success criterion (docs/01
"Success criteria" -- "demonstrates, across multiple seeds, that the full
agent improves feasible-goal completion over a static-policy baseline...
Oracle-feasibility performance defines the headroom") has never actually
been run. Every rigorous multi-seed, bootstrap-CI comparison built so far
(D-042, D-069, D-070--D-078) uses privileged-state feasibility
(`goal_feasible()`), not the real perceptual pipeline (CLIP,
`atr.pipeline.run_end_to_end_episode()`). Every use of the real perceptual
pipeline (D-029, D-054/D-055, D-062, D-064) has been a single episode or a
handful, never passed through `atr.evaluation.harness`'s statistical
machinery. This module closes that gap directly: it runs the real
full-agent pipeline (real language parsing, real CLIP-perceived
feasibility, a trained Q-table decision, real arm motion) across multiple
seeds, paired against `static_policy` (also real arm motion, no perception
at all) and `feasibility_aware_policy` (privileged-state oracle
feasibility -- the headroom reference docs/01 itself names), and reports
real bootstrap confidence intervals via `atr.evaluation.harness.bootstrap_ci()`
(D-042), the same machinery every other comparison in this project uses.

`static_policy` and `feasibility_aware_policy` (`atr.policies.baselines`)
never call `env.render()`, so D-022's rendering-desync bug never applies to
them -- they run in-process across every seed, the same as every existing
privileged-state comparison in this project. The full-agent policy does
render (twice per episode, `run_end_to_end_episode()`'s own verified-safe
budget), so it needs one fresh subprocess per episode
(`run_full_agent_episode_subprocess.py`, D-052's same pattern) -- one
subprocess per seed, never accumulating resets within one process.

Scope, disclosed rather than hidden: this runs against the
ReplicaCAD-Humanoid env's `kitchen_cabinet` scene specifically, the one
CLIP is actually calibrated for (D-020/D-027) -- not a claim about every
env variant or every scene in this project. The Q-table is trained once,
privileged-state, before any seed in the comparison runs (matching
`atr.pipeline`'s own documented split between cheap privileged-state
*training* and real-perception *evaluation*).
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

from atr.envs.tidy_up_env_replicacad_humanoid import replicacad_humanoid_example
from atr.envs.tidy_up_replicacad_humanoid_policies import _TRAY_SLOTS, attempt_goal
from atr.evaluation.harness import bootstrap_ci, run_episode
from atr.language.goal_graph import GoalGraph
from atr.language.instruction_parser import parse_instruction
from atr.pipeline import HUMANOID_OBJECTS, train_q_table_replicacad_humanoid
from atr.policies.baselines import feasibility_aware_policy, static_policy

_SUBPROCESS_SCRIPT = (
    Path(__file__).resolve().parents[1] / "envs" / "run_full_agent_episode_subprocess.py"
)


def instruction_graph() -> GoalGraph:
    """Same construction `atr.pipeline._instruction_graph()` uses -- goal
    ids come from the parser, not the hand-authored example directly, so a
    Q-table trained against this graph has matching keys at evaluation
    time."""
    return parse_instruction(replicacad_humanoid_example().instruction_text, HUMANOID_OBJECTS)


def serialize_q_table(q_table: dict) -> list:
    """JSON has no tuple keys and no non-string dict keys -- `(goal_id,
    feasible) -> {action: q}` becomes a list of records the subprocess
    script's `load_q_table()` reconstructs exactly."""
    return [
        {"goal_id": key[0], "feasible": key[1], "actions": {str(action): q for action, q in actions.items()}}
        for key, actions in q_table.items()
    ]


def _make_env(intervention_kind: str, onset_step_range: tuple[int, int]):
    """render_mode=None -- this factory is only ever used for the two
    privileged-state policies below, which never call env.render(), so
    D-022's rendering-desync bug never applies here."""
    import gymnasium as gym

    return gym.make(
        "TidyUp-ReplicaCAD-Humanoid-v1", num_envs=1, obs_mode="state",
        render_mode=None, sim_backend="physx_cpu", control_mode="pd_joint_pos",
        intervention_kind=intervention_kind, onset_step_range=onset_step_range,
    )


def _run_full_agent_episode_subprocess(
    seed: int, q_table_path: Path, intervention_kind: str, scene_variant: str,
    onset_step_range: tuple[int, int], out_dir: Path,
) -> dict:
    out_path = out_dir / f"full_agent_seed_{seed}.json"
    subprocess.run(
        [
            sys.executable, str(_SUBPROCESS_SCRIPT),
            "--seed", str(seed),
            "--intervention-kind", intervention_kind,
            "--scene-variant", scene_variant,
            "--onset-step-min", str(onset_step_range[0]),
            "--onset-step-max", str(onset_step_range[1]),
            "--q-table-path", str(q_table_path),
            "--out", str(out_path),
        ],
        check=True,
    )
    return json.loads(out_path.read_text())


def run_full_agent_benchmark(
    seeds: list[int],
    q_table: dict | None = None,
    intervention_kind: str = "chef_can_destroyed",
    scene_variant: str = "kitchen_cabinet",
    onset_step_range: tuple[int, int] = (1, 3),
    metrics: tuple[str, ...] = ("goals_achieved", "wasted_steps"),
    n_resamples: int = 2000,
    ci: float = 0.95,
) -> dict[str, dict[str, tuple[float, float, float]]]:
    """Paired, real, multi-seed comparison of `static` (real arm motion, no
    perception), `oracle_feasibility` (privileged-state headroom reference),
    and `full_agent` (real CLIP perception + trained Q-table + real arm
    motion) -- the project's own success criterion, run for the first time.
    Returns `{policy_name: {metric: (mean, lo, hi)}}`, the exact same shape
    `atr.evaluation.harness.compare_policies()` returns, for direct
    consistency with every other comparison in this project.
    """
    if q_table is None:
        q_table = train_q_table_replicacad_humanoid()
    graph = instruction_graph()

    static_episodes = [
        run_episode(
            lambda: _make_env(intervention_kind, onset_step_range),
            lambda env: static_policy(env, graph, attempt_goal, _TRAY_SLOTS),
            seed,
        )
        for seed in seeds
    ]
    oracle_episodes = [
        run_episode(
            lambda: _make_env(intervention_kind, onset_step_range),
            lambda env: feasibility_aware_policy(env, graph, attempt_goal, _TRAY_SLOTS),
            seed,
        )
        for seed in seeds
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        q_table_path = tmp_path / "q_table.json"
        q_table_path.write_text(json.dumps(serialize_q_table(q_table)))
        full_agent_episodes = [
            _run_full_agent_episode_subprocess(
                seed, q_table_path, intervention_kind, scene_variant, onset_step_range, tmp_path,
            )
            for seed in seeds
        ]

    episodes = {
        "static": static_episodes,
        "oracle_feasibility": oracle_episodes,
        "full_agent": full_agent_episodes,
    }
    return {
        name: {
            metric: bootstrap_ci(
                [float(ep[metric]) for ep in name_episodes], n_resamples=n_resamples, ci=ci,
            )
            for metric in metrics
        }
        for name, name_episodes in episodes.items()
    }

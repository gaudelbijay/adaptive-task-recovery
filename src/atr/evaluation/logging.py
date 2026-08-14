"""The log interface docs/03-system-architecture.md's data-flow step 6 has
described since the diagram was first drawn ("Log predictions, decisions,
violations, and oracle labels for analysis") but nothing ever built --
STATUS.md's interfaces row still listed it as not started as of D-055.

Same discipline D-040 used for `AdaptivePolicy`/`EmbodimentInterface`:
don't invent a schema first and check code against it later. Every policy
in `atr.policies.baselines` and every env-specific pipeline
(`atr.pipeline.run_end_to_end_episode`, `dinov2_probe.run_end_to_end_episode_dinov2`)
already produces exactly one shape via `_summarize()`:
`{"per_goal": {goal_id: {"achieved", "steps_used", "skipped", ...}},
"goals_achieved", "total_steps", "wasted_steps", <policy-specific
"*_violated" keys>}`. `build_episode_log()` below is that shape,
losslessly, plus the two things missing from it today: which object each
goal id actually targets (from the `GoalGraph` already passed to every
policy) and the oracle existence label for that object (`env.unwrapped._exists`,
already read directly by every test in this project, never before
attached to a policy's own result). No new fields invented beyond that --
"predictions" (`perceived_feasible`, when a vision-based policy produced
one) and "decisions" (`skipped`/`substitution_attempted`) already exist in
`per_goal`; this only adds "oracle labels" and normalizes "violations"
(any `*_violated` key, wherever a policy happens to add one, rather than
requiring every caller to know their exact names). D-094 also preserves
aggregate navigation replans and safety stops when a policy supplies them."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from atr.language.goal_graph import GoalGraph


def _jsonable(value):
    """Every per_goal outcome in this project can contain numpy scalars
    (e.g. `naive_substitution_policy`'s `goal_achieved()` returns
    `np.bool_`/`np.True_`, confirmed directly while investigating D-055) --
    `json.dumps` rejects those outright, so convert on the way out rather
    than let every caller discover this by hitting a `TypeError`."""
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def build_episode_log(
    result: dict, graph: GoalGraph, oracle_exists: dict[str, bool],
    seed: int | None = None, policy_name: str | None = None,
) -> dict:
    """Combines a policy's own result dict (whatever `_summarize()`-shaped
    dict `static_policy`/`feasibility_aware_policy`/`naive_substitution_policy`/
    `learned_policy`/`run_end_to_end_episode`/`run_end_to_end_episode_dinov2`
    already returns) with the `GoalGraph` it was run against and the
    episode's privileged oracle labels, into one structured record safe to
    persist with `append_episode_log()`. Does not read simulator state
    itself -- `oracle_exists` must be captured by the caller before the env
    closes, same as every test in this project already does directly."""
    goals_by_id = {goal.id: goal for goal in graph.goals}
    per_goal_log = {}
    for goal_id, outcome in result["per_goal"].items():
        goal = goals_by_id.get(goal_id)
        target_object = goal.target_object if goal is not None else None
        per_goal_log[goal_id] = {
            "target_object": target_object,
            "oracle_feasible": oracle_exists.get(target_object) if target_object else None,
            **outcome,
        }
    violations = {key: value for key, value in result.items() if key.endswith("_violated")}
    record = {
        "seed": seed,
        "policy_name": policy_name,
        "instruction_text": graph.instruction_text,
        "per_goal": per_goal_log,
        "goals_achieved": result["goals_achieved"],
        "total_steps": result["total_steps"],
        "wasted_steps": result["wasted_steps"],
        "violations": violations,
    }
    # D-094/D-110: preserve aggregate navigation adaptation/failure metrics
    # when the policy result supplies them, while keeping older/custom result
    # shapes valid and byte-for-byte free of invented navigation values.
    for metric in (
        "navigation_replans",
        "navigation_safety_blocks",
        "navigation_failures",
    ):
        if metric in result:
            record[metric] = result[metric]
    return _jsonable(record)


def append_episode_log(path: str | Path, record: dict) -> None:
    """One JSON object per line (JSONL) so a run's log grows by simple
    appends and partial logs from an interrupted run are still readable --
    unlike a single JSON array, which a crash mid-write would corrupt."""
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")


def read_episode_logs(path: str | Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]

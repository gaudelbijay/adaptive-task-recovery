"""Experiment tracking (D-056 changelog's flagged next step; ai-notes/status.md's
shared row has listed it as not started since D-042/D-043). No new
dependency (wandb/mlflow, a cloud service, a database) -- nothing in this
project's toy-scale, local, single-machine reality justifies one yet, and
those would all be aspirational the same way D-040 already found and
corrected `AdaptivePolicy`'s stateful-class pseudocode to be. What was
actually missing: `compare_policies()` (D-042) and `build_episode_log()`
(D-056) already produce a real bootstrap-CI report and real per-episode
logs, but nothing above them ever persisted *which run* produced them,
when, or against which commit -- every comparison in this project's
history lives only in `ai-notes/decisions.md` prose. `track_comparison()`
below is the thin layer that was missing: run `compare_policies()`, and
also write one `summary.json` (run metadata + the report itself) next to
the per-policy JSONL logs it already knows how to produce, under
`data/runs/` -- gitignored per D-032, same as every other generated
artifact in this project, not committed history.
"""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from atr.evaluation.harness import EnvFactory, PolicyFn, compare_policies
from atr.language.goal_graph import GoalGraph

_REPO_ROOT = Path(__file__).resolve().parents[3]


def _git_commit() -> str | None:
    """Short commit hash of whatever's checked out when a run happens, so
    a later reader can tell which code version a summary.json belongs to
    -- best-effort: a shallow clone or a repo-less environment (e.g. a CI
    artifact download) shouldn't make tracking a run fail outright."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True, cwd=_REPO_ROOT,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def track_comparison(
    run_name: str,
    env_factory: EnvFactory,
    policies: dict[str, PolicyFn],
    seeds: list[int],
    graph: GoalGraph,
    metrics: tuple[str, ...] = ("goals_achieved", "wasted_steps"),
    n_resamples: int = 2000,
    ci: float = 0.95,
    runs_dir: str | Path = "data/runs",
) -> dict:
    """Runs `compare_policies()` (D-042) exactly as before, plus persists
    the run: `{runs_dir}/{run_id}/summary.json` (this function's return
    value) and `{runs_dir}/{run_id}/{policy_name}.jsonl` (per-episode logs,
    D-056, via `compare_policies()`'s own `log_dir`). `run_id` is
    `<UTC timestamp, microsecond precision>_<run_name>`, sortable by
    construction, so `list_runs()` doesn't need to parse anything to order
    runs by time -- microseconds, not just seconds, since two tracked runs
    back-to-back (e.g. a quick toy comparison like this module's own
    tests) can otherwise land in the same second and collide in sort
    order."""
    now = datetime.now(timezone.utc)
    run_id = f"{now.strftime('%Y%m%dT%H%M%S%f')}Z_{run_name}"
    run_dir = Path(runs_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    report = compare_policies(
        env_factory, policies, seeds, metrics=metrics, n_resamples=n_resamples, ci=ci,
        graph=graph, log_dir=str(run_dir),
    )

    summary = {
        "run_id": run_id,
        "run_name": run_name,
        "timestamp": now.isoformat(),
        "git_commit": _git_commit(),
        "seeds": seeds,
        "policy_names": sorted(policies),
        "metrics": list(metrics),
        "n_resamples": n_resamples,
        "ci": ci,
        "instruction_text": graph.instruction_text,
        "report": report,
    }
    with open(run_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def list_runs(runs_dir: str | Path = "data/runs") -> list[dict]:
    """Every tracked run's summary.json, oldest first -- the "queryable
    registry" this data didn't have before, same shape of gap D-044's
    split registry closed for instruction specs. Returns `[]`, not an
    error, if `runs_dir` doesn't exist yet (no run has been tracked)."""
    runs_dir = Path(runs_dir)
    if not runs_dir.exists():
        return []
    summaries = [
        json.loads(path.read_text())
        for path in sorted(runs_dir.glob("*/summary.json"))
    ]
    return sorted(summaries, key=lambda s: s["timestamp"])

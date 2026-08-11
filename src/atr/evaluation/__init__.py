"""Metrics, splits, and counterfactual tests (docs/03's proposed layout).

`harness.py` (D-042) is the first real implementation of
docs/10-evaluation-and-benchmarks.md's "Statistical protocol": paired
episode seeds across methods, bootstrap confidence intervals. Every
policy comparison in this project before this (D-014, D-025, and every
env-variant test) ran a single seed and asserted a point-in-time result
-- real evidence for a toy case, but not what docs/10 itself specifies a
benchmark comparison needs.

`splits.py` (D-044) is the first queryable dataset-split registry --
train/held-out-paraphrase/held-out-composition instruction specs, per
docs/04's explicit "hold out paraphrases and compositions" requirement.
Previously these existed only as literal strings inside
`test_instruction_parser.py`'s test bodies; real, validated evidence,
but not something anything else could enumerate programmatically.

`logging.py` (D-056) is the log interface docs/03-system-architecture.md's
data-flow step 6 described but nothing had implemented -- attaches oracle
labels and a normalized violations dict to the result shape every policy
in `atr.policies.baselines` already produces, and persists it as JSONL.

`tracking.py` (D-057) is experiment tracking on top of `harness.py` and
`logging.py` -- `track_comparison()` runs a `compare_policies()`
comparison and persists a `summary.json` (run metadata + the bootstrap-CI
report) alongside the per-policy episode logs, under `data/runs/`
(gitignored, generated, per D-032). `list_runs()` is the queryable
registry over what's been tracked so far.

`full_agent_benchmark.py` (D-088) runs docs/01's own "Success criteria"
benchmark for the first time: a real, paired, multi-seed comparison of
`static`, `oracle_feasibility`, and the real full-agent pipeline (real
language parsing, real CLIP-perceived feasibility, a trained Q-table, real
arm motion), reusing `harness.py`'s `bootstrap_ci()`. Since the full-agent
policy renders (unlike every other privileged-state comparison in this
project), it runs each episode in a fresh subprocess
(`atr.envs.run_full_agent_episode_subprocess`), one per seed, to respect
D-022's confirmed rendering-desync bug. First real run surfaced an
undiscovered CLIP robustness gap rather than demonstrating the success
criterion outright -- see the module's own docstring and D-088 in
`ai-notes/decisions.md`.
"""

from __future__ import annotations

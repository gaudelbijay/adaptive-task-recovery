---
title: Scaled Experiment Execution
status: active
last_updated: 2026-08-24
---

# Scaled experiment execution

This workflow turns the project's hand-authored comparisons into deterministic,
paired, resumable experiment artifacts. It guarantees validation and
reproducibility properties; it cannot guarantee that a scientific hypothesis
will receive favorable support. Negative results remain results.

## Frozen benchmark

[`configs/benchmark_v1.json`](../configs/benchmark_v1.json) expands to
3,200 cases and 12,800 policy episodes:

- four embodiments/environment families;
- three ReplicaCAD humanoid layouts;
- nominal, irreversible, and reversible changes;
- early and wide intervention-time distributions;
- 100 paired seeds per matrix cell;
- static, oracle-feasibility, guarded-substitution, and unguarded-substitution
  policies.

Every case ID hashes its complete environment/layout/intervention/timing/seed
identity. Every policy runs on the same case IDs. Do not edit the manifest after
launch: edits produce a different fingerprint and output directory.

## Required launch gates

From a clean, committed checkout using the locked ManiSkill environment:

```bash
PYENV_VERSION=.maniskill pytest -q tests/drafts/test_benchmark_suite.py
PYENV_VERSION=.maniskill python scripts/run_benchmark_suite.py \
  --manifest configs/benchmark_smoke.json --preflight
PYENV_VERSION=.maniskill python scripts/run_benchmark_suite.py \
  --manifest configs/benchmark_smoke.json --fail-fast
```

Then exercise one seed in **every** full-matrix cell. This is more meaningful
than testing one arbitrary shard because it touches all environment adapters,
layouts, interventions, and policies:

```bash
python scripts/run_benchmark_suite.py \
  --manifest configs/benchmark_v1.json --pilot --preflight
python scripts/run_benchmark_suite.py \
  --manifest configs/benchmark_v1.json --pilot --fail-fast
```

Do not launch the 12,800 episodes unless the pilot has zero failed records and
its aggregate metrics have been inspected for impossible values.

## SLURM launch and resume

Install the exact project commit and dependencies on every node, ensure the
dataset/assets are visible at identical paths, and use a shared output volume.

```bash
mkdir -p results/slurm
ATR_PYTHON=/path/to/venv/bin/python \
  sbatch --array=0-63 scripts/slurm_benchmark_suite.sh
```

The stable hash assigns each case to one of 64 shards. A shard keeps all
policies for a case together, preserving paired comparisons. Results are one
atomic JSON file per case/policy; re-submitting the identical array resumes and
skips valid completed records. Failed or corrupt records are retried.

## Completion and aggregation

Aggregation rejects missing cases, extra cases, duplicates, absent reference
policies, and unpaired policy results before computing any statistic:

```bash
python scripts/aggregate_benchmark_suite.py \
  --manifest configs/benchmark_v1.json \
  --run-dir results/benchmarks/adaptive_recovery_benchmark_v1__FINGERPRINT
```

Outputs:

- `aggregate.json`: overall and per-environment/layout/condition bootstrap CIs
  plus paired deltas against `oracle_feasibility`;
- `summary_table.csv`: flat overall table for analysis tooling;
- `records/*.json`: raw outcome, metrics, final oracle existence, duration,
  exact case identity, manifest fingerprint, and full git commit.

Preserve the complete run directory as the experiment artifact. Never copy only the
summary table: failure analysis and reproducibility require raw records.

## What this benchmark still does not solve

- Favorable results are not guaranteed and must not be tuned into existence.
- The current policies do not include a competitive external LLM/VLM replanner.
- Manipulation remains abstracted in several environments.
- Seed variation is not a substitute for additional tasks and object sets.
- The third-layout fixed-crop CLIP feasibility method currently fails (D-123).

Those are scientific work packages, not cluster-infrastructure defects. This
runner makes them measurable without silently losing, duplicating, or pooling
the wrong episodes.

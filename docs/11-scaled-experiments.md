---
title: Scaled Experiment Execution
status: active
last_updated: 2026-08-27
---

# Scaled experiment execution

This workflow turns the project's hand-authored comparisons into deterministic,
paired, resumable experiment artifacts. It guarantees validation and
reproducibility properties; it cannot guarantee that a scientific hypothesis
will receive favorable support. Negative results remain results.

## Completed execution and metric audit

The full v1 matrix completed on Jarvis: 3,200 paired cases and 12,800 policy
episodes. Oracle feasibility and static execution have identical overall goal
achievement (1.68625), while static wastes 14.24 additional steps per paired
case (95% bootstrap CI 12.708--15.842). This is evidence for efficiency under
world change, not improved recall.

Do **not** use the v1 or guard-v2 `constraint_violations` columns. Those runs
derived the metric from optional policy-specific result keys, so static and
oracle policies that omitted the key were scored as zero. D-128 corrected the
runner to call the environment's final oracle evaluator for every policy. The
content-addressed v3 safety run (500 cases, 2,000 policy episodes) reports:

- static: 1.000 violations/case;
- oracle feasibility: 0.752 (95% CI 0.714--0.790);
- unguarded substitution: 0.778 (0.742--0.814);
- effect-aware guard: 0.000.

The guard's safety is not free: it achieves 1.00 goals/case versus 1.69 for
oracle feasibility because the fixed bowl skill itself is unsafe and no safe
alternative trajectory exists in the current skill library. Report this as a
safety/recall frontier and an execution-skill limitation.

High-level learned-policy results are stored separately under
`results/rl_training`. They use the abstract teleport-on-success skill contract
and therefore remain decision-layer diagnostics. Continuous, non-teleport
manipulation PPO uses the official ManiSkill three-seed/50M-transition settings
for PickCube, randomized PickSingleYCB, and Unitree G1 apple-in-bowl. Its
independent 256-episode-per-seed evaluation is complete: pooled success is
755/768 (98.31%) for PickCube, 530/768 (69.01%) for PickSingleYCB, and 767/768
(99.87%) for G1 apple-in-bowl. Exact intervals, seed dispersion, simulator
capacity audit, and claim boundaries are in
[`14-results-and-claim-boundaries.md`](14-results-and-claim-boundaries.md).

The newer integrated learned-recovery run is separate from both result sets.
It uses `LearnedRecovery-v1`, continuous Panda joint control, a force-driven
mid-episode intervention, a protected-object constraint, three methods, three
seeds, and 100M requested transitions per seed. Job continuation is automatic:
`scripts/slurm_learned_recovery_ppo.sh` saves atomically on the scheduler's
pre-timeout signal and resubmits only incomplete array elements. Independent
intervention and nominal evaluation is launched through
`scripts/slurm_learned_recovery_eval.sh`; the primary aggregate is
safety-qualified success with branch-specific reporting.

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

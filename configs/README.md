# `configs/`

Versioned experiment manifests consumed by
[`atr.evaluation.benchmark_suite`](../src/atr/evaluation/benchmark_suite.py).

- `benchmark_smoke.json`: four paired cases/eight policy episodes for
  validating a machine before a launch.
- `benchmark_v1.json`: 3,200 deterministic cases and 12,800 paired policy
  episodes across four embodiments, three ReplicaCAD layouts, nominal/
  irreversible/reversible conditions, early and wide intervention timing, and
  100 seeds.

Always run `scripts/run_benchmark_suite.py --preflight` first. Generated data
belongs under `results/` (gitignored), not in source control.

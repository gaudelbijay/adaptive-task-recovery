#!/usr/bin/env bash
# Submit after: mkdir -p results/slurm
# Example: sbatch --array=0-63 scripts/slurm_benchmark_suite.sh
# Re-submit the same array to resume; completed case/policy artifacts are skipped.
# Override paths/count with ATR_MANIFEST, ATR_OUTPUT, ATR_SHARD_COUNT, ATR_PYTHON.
#SBATCH --job-name=atr-benchmark
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=results/slurm/%A_%a.out
#SBATCH --error=results/slurm/%A_%a.err

set -euo pipefail

ATR_MANIFEST="${ATR_MANIFEST:-configs/benchmark_v1.json}"
ATR_OUTPUT="${ATR_OUTPUT:-results/benchmarks}"
ATR_SHARD_COUNT="${ATR_SHARD_COUNT:-64}"
ATR_PYTHON="${ATR_PYTHON:-python}"

test -n "${SLURM_ARRAY_TASK_ID:-}" || {
  echo "SLURM_ARRAY_TASK_ID is missing; submit this script as an array" >&2
  exit 2
}

"${ATR_PYTHON}" scripts/run_benchmark_suite.py \
  --manifest "${ATR_MANIFEST}" \
  --output "${ATR_OUTPUT}" \
  --shard-index "${SLURM_ARRAY_TASK_ID}" \
  --shard-count "${ATR_SHARD_COUNT}"

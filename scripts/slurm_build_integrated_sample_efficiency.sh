#!/usr/bin/env bash
#SBATCH --job-name=atr-sample-accounting
#SBATCH --partition=compute-short
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=results/slurm/sample_accounting_%j.out
#SBATCH --error=results/slurm/sample_accounting_%j.err

set -euo pipefail
: "${ATR_PERFORMANCE_REPORT:?set ATR_PERFORMANCE_REPORT}"
: "${ATR_METHOD_CONTRACT:?set ATR_METHOD_CONTRACT}"
: "${ATR_SAMPLE_ACCOUNTING_PREFIX:?set ATR_SAMPLE_ACCOUNTING_PREFIX}"
.venv/bin/python scripts/build_integrated_sample_efficiency.py \
  --performance "${ATR_PERFORMANCE_REPORT}" \
  --method-contract "${ATR_METHOD_CONTRACT}" \
  --output-prefix "${ATR_SAMPLE_ACCOUNTING_PREFIX}"

#!/usr/bin/env bash
#SBATCH --job-name=atr-strict-aggregate
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=results/slurm/strict_removal_aggregate_%j.out
#SBATCH --error=results/slurm/strict_removal_aggregate_%j.err

set -euo pipefail
mkdir -p results/slurm
ATR_STRICT_COMPARISON_CONFIG="${ATR_STRICT_COMPARISON_CONFIG:-configs/strict_removal_comparison_v1.json}"
.venv/bin/python scripts/aggregate_strict_removal_comparison.py \
  --config "${ATR_STRICT_COMPARISON_CONFIG}"

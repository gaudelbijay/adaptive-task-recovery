#!/usr/bin/env bash
#SBATCH --job-name=atr-temporal-effect
#SBATCH --partition=compute-short
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=results/slurm/temporal_effect_%j.out
#SBATCH --error=results/slurm/temporal_effect_%j.err

set -euo pipefail
: "${ATR_TEMPORAL_ABLATION_CONFIG:?set ATR_TEMPORAL_ABLATION_CONFIG}"
: "${ATR_TEMPORAL_ABLATION_OUTPUT:?set ATR_TEMPORAL_ABLATION_OUTPUT}"
.venv/bin/python scripts/compare_temporal_ssl_continuation.py \
  --config "${ATR_TEMPORAL_ABLATION_CONFIG}" \
  --output "${ATR_TEMPORAL_ABLATION_OUTPUT}"

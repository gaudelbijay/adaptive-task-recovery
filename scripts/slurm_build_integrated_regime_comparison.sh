#!/usr/bin/env bash
#SBATCH --job-name=atr-integrated-figure
#SBATCH --partition=compute-short
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=results/slurm/integrated_figure_%j.out
#SBATCH --error=results/slurm/integrated_figure_%j.err

set -euo pipefail
: "${ATR_INTEGRATED_CONFIG:?set ATR_INTEGRATED_CONFIG}"
: "${ATR_INTEGRATED_PREFIX:?set ATR_INTEGRATED_PREFIX}"
.venv/bin/python scripts/build_integrated_regime_comparison.py \
  --config "${ATR_INTEGRATED_CONFIG}" --output-prefix "${ATR_INTEGRATED_PREFIX}"

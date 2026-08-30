#!/usr/bin/env bash
#SBATCH --job-name=atr-integrated-select
#SBATCH --partition=compute-short
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --output=results/slurm/integrated_select_%j.out
#SBATCH --error=results/slurm/integrated_select_%j.err

set -euo pipefail
: "${ATR_SELECTION_CONFIG:?set ATR_SELECTION_CONFIG}"
: "${ATR_SELECTION_OUTPUT:?set ATR_SELECTION_OUTPUT}"
mkdir -p results/slurm
.venv/bin/python scripts/select_integrated_visual_policy.py \
  --config "${ATR_SELECTION_CONFIG}" --output "${ATR_SELECTION_OUTPUT}"

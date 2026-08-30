#!/usr/bin/env bash
#SBATCH --job-name=atr-visual-select
#SBATCH --partition=compute-short
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=results/slurm/visual_select_%j.out
#SBATCH --error=results/slurm/visual_select_%j.err

set -euo pipefail
mkdir -p results/slurm
"${ATR_PYTHON:-.venv/bin/python}" scripts/select_visual_recovery_initialization.py \
  --config "${ATR_SELECTION_CONFIG:-configs/visual_recovery_selection_v1.json}" \
  --output "${ATR_SELECTION_OUTPUT:-results/visual_recovery_ppo/visual_recovery_selection_v1}"

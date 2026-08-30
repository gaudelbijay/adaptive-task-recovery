#!/usr/bin/env bash
#SBATCH --job-name=atr-visual-compare
#SBATCH --partition=compute-short
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=results/slurm/visual_compare_%j.out
#SBATCH --error=results/slurm/visual_compare_%j.err

set -euo pipefail
mkdir -p results/slurm
"${ATR_PYTHON:-.venv/bin/python}" scripts/compare_visual_recovery_candidates.py \
  --config "${ATR_COMPARISON_CONFIG:-configs/visual_recovery_comparison_v1.json}" \
  --output "${ATR_COMPARISON_OUTPUT:-results/final_visual_comparison}"

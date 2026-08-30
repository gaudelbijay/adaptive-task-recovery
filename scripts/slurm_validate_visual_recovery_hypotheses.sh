#!/usr/bin/env bash
#SBATCH --job-name=atr-v3-hypotheses
#SBATCH --partition=compute-short
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=results/slurm/visual_hypotheses_%j.out
#SBATCH --error=results/slurm/visual_hypotheses_%j.err

set -euo pipefail
ATR_HYPOTHESIS_CONFIG="${ATR_HYPOTHESIS_CONFIG:-configs/visual_recovery_hypothesis_validation_v1.json}"
ATR_HYPOTHESIS_OUTPUT="${ATR_HYPOTHESIS_OUTPUT:-results/final_visual_comparison}"
ATR_PYTHON="${ATR_PYTHON:-.venv/bin/python}"
"${ATR_PYTHON}" scripts/validate_visual_recovery_hypotheses.py \
  --config "${ATR_HYPOTHESIS_CONFIG}" --output "${ATR_HYPOTHESIS_OUTPUT}"

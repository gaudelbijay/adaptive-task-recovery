#!/usr/bin/env bash
#SBATCH --job-name=atr-visual-aggregate
#SBATCH --partition=compute-short
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=results/slurm/visual_aggregate_%j.out
#SBATCH --error=results/slurm/visual_aggregate_%j.err

set -euo pipefail
ATR_VISUAL_CONFIG="${ATR_VISUAL_CONFIG:-configs/visual_recovery_ppo_gate_v1.json}"
ATR_VISUAL_OUTPUT="${ATR_VISUAL_OUTPUT:-results/visual_recovery_ppo}"
ATR_PYTHON="${ATR_PYTHON:-.venv/bin/python}"
"${ATR_PYTHON}" scripts/aggregate_visual_recovery.py \
  --config "${ATR_VISUAL_CONFIG}" --output "${ATR_VISUAL_OUTPUT}"

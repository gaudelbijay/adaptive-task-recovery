#!/usr/bin/env bash
#SBATCH --job-name=atr-v4-physics
#SBATCH --partition=gpu-l40s
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=results/slurm/v4_physics_%j.out
#SBATCH --error=results/slurm/v4_physics_%j.err

set -euo pipefail
mkdir -p results/slurm results
export PYTHONUNBUFFERED=1
.venv/bin/python scripts/evaluate_v4_intervention_reliability.py \
  --output results/v4_intervention_reliability_v4.json

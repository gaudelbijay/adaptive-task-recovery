#!/usr/bin/env bash
#SBATCH --job-name=atr-v4-dino-state
#SBATCH --partition=gpu-l40s
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=results/slurm/v4_dino_state_%j.out
#SBATCH --error=results/slurm/v4_dino_state_%j.err

set -euo pipefail
mkdir -p results/slurm results/v4_dino_state_estimator_pilot
export PYTHONUNBUFFERED=1
.venv/bin/python scripts/train_v4_dino_state_estimator.py

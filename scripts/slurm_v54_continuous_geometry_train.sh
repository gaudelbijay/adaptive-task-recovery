#!/usr/bin/env bash
#SBATCH --job-name=atr-v54-geometry
#SBATCH --partition=gpu-l40s
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=110G
#SBATCH --time=04:00:00
#SBATCH --output=results/slurm/v54_geometry_%j.out
#SBATCH --error=results/slurm/v54_geometry_%j.err
set -euo pipefail
: "${ATR_VISUAL_CONFIG:?set ATR_VISUAL_CONFIG}"
mkdir -p results/slurm; export PYTHONUNBUFFERED=1
.venv/bin/python scripts/train_v54_continuous_geometry.py --config "$ATR_VISUAL_CONFIG" --output "${ATR_VISUAL_OUTPUT:-results/visual_recovery_ppo}" --task-index 0

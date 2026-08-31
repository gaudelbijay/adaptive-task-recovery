#!/usr/bin/env bash
#SBATCH --job-name=atr-v55-router
#SBATCH --partition=gpu-l40s
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --output=results/slurm/v55_router_%j.out
#SBATCH --error=results/slurm/v55_router_%j.err
set -euo pipefail
: "${ATR_VISUAL_CONFIG:?set ATR_VISUAL_CONFIG}"
mkdir -p results/slurm;export PYTHONUNBUFFERED=1
.venv/bin/python scripts/train_v55_binary_geometry_router.py --config "$ATR_VISUAL_CONFIG" --output "${ATR_VISUAL_OUTPUT:-results/visual_recovery_ppo}" --task-index 0

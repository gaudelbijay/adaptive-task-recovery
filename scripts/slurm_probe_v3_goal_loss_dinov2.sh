#!/usr/bin/env bash
#SBATCH --job-name=atr-dino-goal-loss
#SBATCH --partition=gpu-l40s
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=results/slurm/dino_goal_loss_%j.out
#SBATCH --error=results/slurm/dino_goal_loss_%j.err

set -euo pipefail
mkdir -p results/slurm results/probes
export PYTHONUNBUFFERED=1
.venv/bin/python scripts/probe_v3_goal_loss_dinov2.py \
  --output results/probes/v3_goal_loss_dinov2_v3.json

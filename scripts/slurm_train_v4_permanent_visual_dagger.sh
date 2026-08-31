#!/usr/bin/env bash
#SBATCH --job-name=atr-v4-perm-vis
#SBATCH --partition=gpu-l40s
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=results/slurm/v4_perm_visual_%j.out
#SBATCH --error=results/slurm/v4_perm_visual_%j.err

set -euo pipefail
mkdir -p results/slurm results/visual_recovery_ppo
export PYTHONUNBUFFERED=1
.venv/bin/python scripts/train_v4_permanent_visual_dagger.py \
  --config configs/visual_recovery_v4_permanent_dagger_pilot.json \
  --output results/visual_recovery_ppo

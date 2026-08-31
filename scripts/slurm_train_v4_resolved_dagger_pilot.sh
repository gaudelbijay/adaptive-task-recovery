#!/usr/bin/env bash
#SBATCH --job-name=atr-v4-dagger
#SBATCH --partition=gpu-l40s
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=results/slurm/v4_dagger_%j.out
#SBATCH --error=results/slurm/v4_dagger_%j.err

set -euo pipefail
mkdir -p results/slurm results/visual_recovery_v4
export PYTHONUNBUFFERED=1
.venv/bin/python scripts/train_v4_resolved_progress_dagger.py \
  --config configs/visual_recovery_v4_resolved_dagger_pilot.json \
  --output results/visual_recovery_v4

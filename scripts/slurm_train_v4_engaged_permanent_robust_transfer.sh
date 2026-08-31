#!/usr/bin/env bash
#SBATCH --job-name=atr-v4-engaged-robust
#SBATCH --partition=gpu-l40s
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=results/slurm/v4_engaged_robust_%j.out
#SBATCH --error=results/slurm/v4_engaged_robust_%j.err

set -euo pipefail
mkdir -p results/slurm
export PYTHONUNBUFFERED=1
.venv/bin/python scripts/train_manipulation_ppo.py \
  --config configs/learned_recovery_v4_engaged_permanent_robust_transfer.json \
  --task-index 0

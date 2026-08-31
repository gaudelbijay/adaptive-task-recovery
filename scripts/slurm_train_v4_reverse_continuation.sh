#!/usr/bin/env bash
#SBATCH --job-name=atr-v4-reverse-cont
#SBATCH --partition=gpu-l40s
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:40:00
#SBATCH --array=0-1
#SBATCH --output=results/slurm/v4_reverse_cont_%A_%a.out
#SBATCH --error=results/slurm/v4_reverse_cont_%A_%a.err

set -euo pipefail
mkdir -p results/slurm
export PYTHONUNBUFFERED=1
.venv/bin/python scripts/train_manipulation_ppo.py \
  --config configs/learned_recovery_v4_reverse_continuation.json \
  --task-index "${SLURM_ARRAY_TASK_ID}"

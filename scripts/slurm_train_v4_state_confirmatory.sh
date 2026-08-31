#!/usr/bin/env bash
#SBATCH --job-name=atr-v4-confirm
#SBATCH --partition=gpu-l40s
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --array=0-3
#SBATCH --output=results/slurm/v4_confirm_%A_%a.out
#SBATCH --error=results/slurm/v4_confirm_%A_%a.err

set -euo pipefail
mkdir -p results/slurm results/learned_recovery_v4
.venv/bin/python scripts/train_manipulation_ppo.py \
  --config configs/learned_recovery_v4_state_confirmatory.json \
  --output results/learned_recovery_v4 \
  --task-index "${SLURM_ARRAY_TASK_ID}"

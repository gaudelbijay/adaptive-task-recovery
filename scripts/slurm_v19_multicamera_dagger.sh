#!/usr/bin/env bash
#SBATCH --job-name=atr-v19-multicamera-dagger
#SBATCH --partition=gpu-l40s
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=80G
#SBATCH --time=06:00:00
#SBATCH --output=results/slurm/v19_multicamera_dagger_%A_%a.out
#SBATCH --error=results/slurm/v19_multicamera_dagger_%A_%a.err

set -euo pipefail
: "${ATR_MULTICAMERA_DAGGER_CONFIG:?set ATR_MULTICAMERA_DAGGER_CONFIG}"
mkdir -p results/slurm
export PYTHONUNBUFFERED=1
.venv/bin/python scripts/train_v19_multicamera_dagger.py \
  --config "${ATR_MULTICAMERA_DAGGER_CONFIG}" \
  --output "${ATR_VISUAL_OUTPUT:-results/visual_recovery_ppo}" \
  --task-index "${SLURM_ARRAY_TASK_ID:-0}"

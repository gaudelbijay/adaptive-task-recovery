#!/usr/bin/env bash
#SBATCH --job-name=atr-v37-dense-canonical
#SBATCH --partition=gpu-l40s
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=100G
#SBATCH --time=04:00:00
#SBATCH --output=results/slurm/v37_dense_canonical_%A_%a.out
#SBATCH --error=results/slurm/v37_dense_canonical_%A_%a.err

set -euo pipefail
: "${ATR_VISUAL_CONFIG:?set ATR_VISUAL_CONFIG}"
mkdir -p results/slurm
export PYTHONUNBUFFERED=1
.venv/bin/python scripts/train_v37_dense_canonical.py \
  --config "${ATR_VISUAL_CONFIG}" --output "${ATR_VISUAL_OUTPUT:-results/visual_recovery_ppo}" \
  --task-index "${SLURM_ARRAY_TASK_ID:-0}"

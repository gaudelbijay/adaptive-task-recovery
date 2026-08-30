#!/usr/bin/env bash
#SBATCH --job-name=atr-v19-canonical-view
#SBATCH --partition=gpu-l40s
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=results/slurm/v19_canonical_view_%A_%a.out
#SBATCH --error=results/slurm/v19_canonical_view_%A_%a.err

set -euo pipefail
: "${ATR_CANONICAL_VIEW_CONFIG:?set ATR_CANONICAL_VIEW_CONFIG}"
mkdir -p results/slurm
export PYTHONUNBUFFERED=1
.venv/bin/python scripts/train_v19_canonical_view.py \
  --config "${ATR_CANONICAL_VIEW_CONFIG}" \
  --output "${ATR_VISUAL_OUTPUT:-results/visual_recovery_ppo}" \
  --task-index "${SLURM_ARRAY_TASK_ID:-0}"

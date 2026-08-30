#!/usr/bin/env bash
#SBATCH --job-name=atr-v19-robust-distill
#SBATCH --partition=gpu-l40s
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=02:00:00
#SBATCH --signal=USR1@300
#SBATCH --output=results/slurm/v19_robust_distill_%A_%a.out
#SBATCH --error=results/slurm/v19_robust_distill_%A_%a.err

set -euo pipefail
: "${ATR_ROBUST_DISTILL_CONFIG:?set ATR_ROBUST_DISTILL_CONFIG}"
ATR_VISUAL_OUTPUT="${ATR_VISUAL_OUTPUT:-results/visual_recovery_ppo}"
mkdir -p results/slurm
export PYTHONUNBUFFERED=1
.venv/bin/python scripts/train_v19_robust_distillation.py \
  --config "${ATR_ROBUST_DISTILL_CONFIG}" \
  --output "${ATR_VISUAL_OUTPUT}" \
  --task-index "${SLURM_ARRAY_TASK_ID:-0}"

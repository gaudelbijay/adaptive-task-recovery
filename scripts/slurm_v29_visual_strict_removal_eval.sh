#!/usr/bin/env bash
#SBATCH --job-name=atr-v29-strict-removal
#SBATCH --partition=gpu-l40s
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=results/slurm/v29_strict_eval_%A_%a.out
#SBATCH --error=results/slurm/v29_strict_eval_%A_%a.err

set -euo pipefail
: "${ATR_VISUAL_CONFIG:?set ATR_VISUAL_CONFIG}"
mkdir -p results/slurm
export PYTHONUNBUFFERED=1
.venv/bin/python scripts/evaluate_v29_visual_recovery_strict_removal.py \
  --config "${ATR_VISUAL_CONFIG}" \
  --strict-config "${ATR_STRICT_CONFIG:-configs/visual_recovery_strict_removal_eval_v1.json}" \
  --output "${ATR_VISUAL_OUTPUT:-results/visual_recovery_ppo}" \
  --task-index "${SLURM_ARRAY_TASK_ID:-0}" \
  --episodes "${ATR_STRICT_EPISODES:-256}" --num-envs "${ATR_STRICT_NUM_ENVS:-32}"

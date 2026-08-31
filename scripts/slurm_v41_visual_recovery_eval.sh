#!/usr/bin/env bash
#SBATCH --job-name=atr-v41-visual-eval
#SBATCH --partition=gpu-l40s
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=results/slurm/v41_visual_eval_%A_%a.out
#SBATCH --error=results/slurm/v41_visual_eval_%A_%a.err

set -euo pipefail
: "${ATR_VISUAL_CONFIG:?set ATR_VISUAL_CONFIG}"
mkdir -p results/slurm
export PYTHONUNBUFFERED=1
for condition in ${ATR_EVAL_CONDITIONS:-nominal intervention}; do
  .venv/bin/python scripts/evaluate_v41_visual_recovery.py \
    --config "${ATR_VISUAL_CONFIG}" \
    --output "${ATR_VISUAL_OUTPUT:-results/visual_recovery_ppo}" \
    --task-index "${SLURM_ARRAY_TASK_ID:-0}" \
    --episodes "${ATR_EVAL_EPISODES:-256}" \
    --num-envs "${ATR_EVAL_NUM_ENVS:-32}" --condition "${condition}"
done

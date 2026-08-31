#!/usr/bin/env bash
#SBATCH --job-name=atr-v60-strict
#SBATCH --partition=gpu-l40s
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=results/slurm/v60_strict_%A_%a.out
#SBATCH --error=results/slurm/v60_strict_%A_%a.err

set -euo pipefail
mkdir -p results/slurm
export PYTHONUNBUFFERED=1
.venv/bin/python scripts/evaluate_v60_visual_recovery_strict_removal.py \
  --config configs/visual_recovery_v60_three_seed.json \
  --strict-config configs/visual_recovery_strict_removal_eval_v1.json \
  --output results/visual_recovery_ppo \
  --task-index "${SLURM_ARRAY_TASK_ID:-0}" \
  --episodes 256 --num-envs 32

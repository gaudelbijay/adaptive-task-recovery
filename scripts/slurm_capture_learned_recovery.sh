#!/usr/bin/env bash
#SBATCH --job-name=atr-recovery-video
#SBATCH --partition=gpu-l40s
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=results/slurm/recovery_video_%A_%a.out
#SBATCH --error=results/slurm/recovery_video_%A_%a.err

set -euo pipefail

ATR_NODE_CACHE="${SLURM_TMPDIR:-/tmp}/atr-recovery-video-${SLURM_JOB_ID}-${SLURM_ARRAY_TASK_ID}"
mkdir -p "${ATR_NODE_CACHE}/gl" "${ATR_NODE_CACHE}/cuda" "${ATR_NODE_CACHE}/xdg"
export __GL_SHADER_DISK_CACHE_PATH="${ATR_NODE_CACHE}/gl"
export CUDA_CACHE_PATH="${ATR_NODE_CACHE}/cuda"
export XDG_CACHE_HOME="${ATR_NODE_CACHE}/xdg"
export PYTHONUNBUFFERED=1

case "${SLURM_ARRAY_TASK_ID}" in
  0) ATR_BRANCH=first_goal_removed ;;
  1) ATR_BRANCH=second_goal_removed ;;
  2) ATR_BRANCH=nominal ;;
  *) echo "invalid array index: ${SLURM_ARRAY_TASK_ID}" >&2; exit 2 ;;
esac

.venv/bin/python scripts/capture_learned_recovery_policy.py \
  --config configs/learned_recovery_ppo_v6.json \
  --task-index "${ATR_CAPTURE_TASK_INDEX:-0}" \
  --branch "${ATR_BRANCH}"

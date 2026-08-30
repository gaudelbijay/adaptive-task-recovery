#!/usr/bin/env bash
#SBATCH --job-name=atr-v3-strict-removal
#SBATCH --partition=gpu-l40s
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=results/slurm/visual_strict_eval_%A_%a.out
#SBATCH --error=results/slurm/visual_strict_eval_%A_%a.err

set -euo pipefail
ATR_NODE_CACHE="${SLURM_TMPDIR:-/tmp}/atr-strict-eval-${SLURM_JOB_ID}-${SLURM_ARRAY_TASK_ID}"
mkdir -p "${ATR_NODE_CACHE}/gl" "${ATR_NODE_CACHE}/cuda" "${ATR_NODE_CACHE}/xdg" results/slurm
export __GL_SHADER_DISK_CACHE_PATH="${ATR_NODE_CACHE}/gl"
export CUDA_CACHE_PATH="${ATR_NODE_CACHE}/cuda"
export XDG_CACHE_HOME="${ATR_NODE_CACHE}/xdg"
export PYTHONUNBUFFERED=1

: "${ATR_VISUAL_CONFIG:?set ATR_VISUAL_CONFIG}"
ATR_STRICT_CONFIG="${ATR_STRICT_CONFIG:-configs/visual_recovery_strict_removal_eval_v1.json}"
ATR_VISUAL_OUTPUT="${ATR_VISUAL_OUTPUT:-results/visual_recovery_ppo}"
ATR_PYTHON="${ATR_PYTHON:-.venv/bin/python}"

"${ATR_PYTHON}" scripts/evaluate_visual_recovery_strict_removal.py \
  --config "${ATR_VISUAL_CONFIG}" \
  --strict-config "${ATR_STRICT_CONFIG}" \
  --output "${ATR_VISUAL_OUTPUT}" \
  --task-index "${SLURM_ARRAY_TASK_ID}" \
  --episodes "${ATR_STRICT_EPISODES:-256}" \
  --num-envs "${ATR_STRICT_NUM_ENVS:-32}"

#!/usr/bin/env bash
#SBATCH --job-name=atr-visual-eval
#SBATCH --partition=gpu-l40s
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=results/slurm/visual_eval_%A_%a.out
#SBATCH --error=results/slurm/visual_eval_%A_%a.err

set -euo pipefail

ATR_VISUAL_CONFIG="${ATR_VISUAL_CONFIG:-configs/visual_recovery_ppo_gate_v1.json}"
ATR_VISUAL_OUTPUT="${ATR_VISUAL_OUTPUT:-results/visual_recovery_ppo}"
ATR_PYTHON="${ATR_PYTHON:-.venv/bin/python}"
ATR_TASK_INDEX="${SLURM_ARRAY_TASK_ID:-0}"
ATR_NODE_CACHE="${SLURM_TMPDIR:-/tmp}/atr-visual-eval-${SLURM_JOB_ID}-${ATR_TASK_INDEX}"
mkdir -p "${ATR_NODE_CACHE}/gl" "${ATR_NODE_CACHE}/cuda" "${ATR_NODE_CACHE}/xdg" results/slurm
export __GL_SHADER_DISK_CACHE_PATH="${ATR_NODE_CACHE}/gl"
export CUDA_CACHE_PATH="${ATR_NODE_CACHE}/cuda"
export XDG_CACHE_HOME="${ATR_NODE_CACHE}/xdg"
export PYTHONUNBUFFERED=1

ATR_EVAL_CONDITIONS="${ATR_EVAL_CONDITIONS:-nominal intervention}"
for ATR_CONDITION in ${ATR_EVAL_CONDITIONS}; do
  "${ATR_PYTHON}" scripts/evaluate_visual_recovery_ppo.py \
    --config "${ATR_VISUAL_CONFIG}" --output "${ATR_VISUAL_OUTPUT}" \
    --task-index "${ATR_TASK_INDEX}" --episodes "${ATR_EVAL_EPISODES:-256}" \
    --num-envs "${ATR_EVAL_NUM_ENVS:-32}" --condition "${ATR_CONDITION}" \
    --progress-head-mode "${ATR_PROGRESS_HEAD_MODE:-normal}" \
    --visual-perturbation "${ATR_VISUAL_PERTURBATION:-none}" \
    --environment-profile "${ATR_ENVIRONMENT_PROFILE:-nominal}"
done

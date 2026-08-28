#!/usr/bin/env bash
#SBATCH --job-name=atr-ne-eval
#SBATCH --partition=gpu-l40s
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=results/slurm/ne_eval_%A_%a.out
#SBATCH --error=results/slurm/ne_eval_%A_%a.err

set -euo pipefail

ATR_NE_CONFIG="${ATR_NE_CONFIG:-configs/learned_recovery_nedreamer_pilot.json}"
ATR_NE_OUTPUT="${ATR_NE_OUTPUT:-results/vision_nedreamer}"
ATR_NE_UPSTREAM="${ATR_NE_UPSTREAM:-/home/bgaudel/.cache/atr/nedreamer-7ca2193}"
ATR_PYTHON="${ATR_PYTHON:-.venv/bin/python}"
ATR_EVAL_EPISODES="${ATR_EVAL_EPISODES:-256}"
ATR_TASK_INDEX="${SLURM_ARRAY_TASK_ID:-0}"
ATR_NODE_CACHE="${SLURM_TMPDIR:-/tmp}/atr-ne-eval-${SLURM_JOB_ID}-${ATR_TASK_INDEX}"
mkdir -p "${ATR_NODE_CACHE}/gl" "${ATR_NODE_CACHE}/cuda" "${ATR_NODE_CACHE}/xdg" results/slurm
export __GL_SHADER_DISK_CACHE_PATH="${ATR_NODE_CACHE}/gl"
export CUDA_CACHE_PATH="${ATR_NODE_CACHE}/cuda"
export XDG_CACHE_HOME="${ATR_NODE_CACHE}/xdg"
export PYTHONUNBUFFERED=1
export WANDB_MODE=disabled
export WANDB_SILENT=true

"${ATR_PYTHON}" scripts/train_vision_nedreamer.py \
  --config "${ATR_NE_CONFIG}" \
  --output "${ATR_NE_OUTPUT}" \
  --task-index "${ATR_TASK_INDEX}" \
  --upstream "${ATR_NE_UPSTREAM}" \
  --eval-only \
  --eval-episodes "${ATR_EVAL_EPISODES}"

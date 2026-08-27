#!/usr/bin/env bash
#SBATCH --job-name=atr-manip-video
#SBATCH --partition=gpu-l40s
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=results/slurm/manip_video_%A_%a.out
#SBATCH --error=results/slurm/manip_video_%A_%a.err

set -euo pipefail

ATR_NODE_CACHE="${SLURM_TMPDIR:-/tmp}/atr-video-${SLURM_JOB_ID}-${SLURM_ARRAY_TASK_ID}"
mkdir -p "${ATR_NODE_CACHE}/gl" "${ATR_NODE_CACHE}/cuda" "${ATR_NODE_CACHE}/xdg"
export __GL_SHADER_DISK_CACHE_PATH="${ATR_NODE_CACHE}/gl"
export CUDA_CACHE_PATH="${ATR_NODE_CACHE}/cuda"
export XDG_CACHE_HOME="${ATR_NODE_CACHE}/xdg"
export PYTHONUNBUFFERED=1

.venv/bin/python scripts/capture_manipulation_policy.py \
  --task-index "${SLURM_ARRAY_TASK_ID}"

#!/usr/bin/env bash
#SBATCH --job-name=atr-v39-opened
#SBATCH --partition=gpu-l40s
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=results/slurm/v39_opened_%A_%a.out
#SBATCH --error=results/slurm/v39_opened_%A_%a.err
set -euo pipefail
: "${ATR_V39_OPENED_CONFIG:?set ATR_V39_OPENED_CONFIG}"
mkdir -p results/slurm; export PYTHONUNBUFFERED=1
.venv/bin/python scripts/run_v39_on_v52_opened_domains.py --config "$ATR_V39_OPENED_CONFIG" --output "${ATR_VISUAL_OUTPUT:-results/visual_recovery_ppo}" --task-index "${SLURM_ARRAY_TASK_ID:-0}"

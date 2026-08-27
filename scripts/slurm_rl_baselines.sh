#!/usr/bin/env bash
#SBATCH --job-name=atr-rl-base
#SBATCH --partition=compute-v2
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=24:00:00
#SBATCH --output=results/slurm/base_%A_%a.out
#SBATCH --error=results/slurm/base_%A_%a.err

set -euo pipefail

ATR_BASELINE_CONFIG="${ATR_BASELINE_CONFIG:-configs/rl_baselines_v1.json}"
ATR_RL_OUTPUT="${ATR_RL_OUTPUT:-results/rl_training}"
ATR_PYTHON="${ATR_PYTHON:-.venv/bin/python}"

"${ATR_PYTHON}" scripts/train_rl_baseline.py \
  --config "${ATR_BASELINE_CONFIG}" \
  --output "${ATR_RL_OUTPUT}" \
  --task-index "${SLURM_ARRAY_TASK_ID}"

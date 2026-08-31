#!/usr/bin/env bash
#SBATCH --job-name=atr-v60-full-stage
#SBATCH --partition=gpu-l40s
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=110G
#SBATCH --time=04:00:00
#SBATCH --output=results/slurm/v60_full_%x_%A_%a.out
#SBATCH --error=results/slurm/v60_full_%x_%A_%a.err
set -euo pipefail
: "${ATR_PIPELINE_STAGE:?set ATR_PIPELINE_STAGE}"
mkdir -p results/slurm; export PYTHONUNBUFFERED=1
.venv/bin/python scripts/run_seeded_config_variant.py \
  --pipeline-config configs/v60_three_seed_pipeline_v1.json \
  --stage "$ATR_PIPELINE_STAGE" --task-index "${SLURM_ARRAY_TASK_ID:-0}" \
  --output "${ATR_VISUAL_OUTPUT:-results/visual_recovery_ppo}"

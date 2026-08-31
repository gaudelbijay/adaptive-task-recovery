#!/usr/bin/env bash
#SBATCH --job-name=atr-v56-stage
#SBATCH --partition=compute-short
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=results/slurm/v56_stage_%j.out
#SBATCH --error=results/slurm/v56_stage_%j.err
set -euo pipefail
mkdir -p results/slurm
.venv/bin/python scripts/build_v56_router_free_checkpoint.py \
  --config configs/visual_recovery_v56_router_free_smoke.json \
  --output "${ATR_VISUAL_OUTPUT:-results/visual_recovery_ppo}"

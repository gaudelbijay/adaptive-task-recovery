#!/usr/bin/env bash
#SBATCH --job-name=atr-v3-five-seed
#SBATCH --partition=compute-short
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=results/slurm/visual_five_seed_%j.out
#SBATCH --error=results/slurm/visual_five_seed_%j.err

set -euo pipefail
mkdir -p results/slurm results/final_visual_comparison
.venv/bin/python scripts/aggregate_five_seed_visual_confirmation.py \
  --config configs/visual_recovery_five_seed_confirmation_v1.json \
  --output results/final_visual_comparison

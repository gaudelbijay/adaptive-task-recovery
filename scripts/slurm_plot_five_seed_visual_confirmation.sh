#!/usr/bin/env bash
#SBATCH --job-name=atr-v3-five-seed-plot
#SBATCH --partition=compute-short
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=results/slurm/visual_five_seed_plot_%j.out
#SBATCH --error=results/slurm/visual_five_seed_plot_%j.err

set -euo pipefail
mkdir -p results/slurm media/results
export MPLBACKEND=Agg
.venv/bin/python scripts/plot_five_seed_visual_confirmation.py \
  --input results/final_visual_comparison/five_seed_confirmation.json \
  --output media/results/v3_visual_recovery_five_seed.png

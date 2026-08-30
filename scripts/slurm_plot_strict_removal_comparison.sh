#!/usr/bin/env bash
#SBATCH --job-name=atr-strict-figure
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=results/slurm/strict_removal_figure_%j.out
#SBATCH --error=results/slurm/strict_removal_figure_%j.err

set -euo pipefail
mkdir -p results/slurm
: "${ATR_STRICT_AGGREGATE:?set ATR_STRICT_AGGREGATE}"
ATR_STRICT_FIGURE_PREFIX="${ATR_STRICT_FIGURE_PREFIX:-results/paper/strict_removal_comparison}"
.venv/bin/python scripts/plot_strict_removal_comparison.py \
  --aggregate "${ATR_STRICT_AGGREGATE}" \
  --output-prefix "${ATR_STRICT_FIGURE_PREFIX}"

#!/usr/bin/env bash
#SBATCH --job-name=atr-v3-result-plot
#SBATCH --partition=compute-short
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=results/slurm/visual_result_plot_%j.out
#SBATCH --error=results/slurm/visual_result_plot_%j.err

set -euo pipefail
ATR_PYTHON="${ATR_PYTHON:-.venv/bin/python}"
"${ATR_PYTHON}" scripts/plot_visual_recovery_results.py \
  --comparison "${ATR_COMPARISON_JSON:-results/final_visual_comparison/comparison.json}" \
  --output-prefix "${ATR_RESULT_FIGURE_PREFIX:-media/results/v3_visual_recovery_comparison}"

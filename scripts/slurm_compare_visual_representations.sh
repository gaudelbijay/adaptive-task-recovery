#!/usr/bin/env bash
#SBATCH --job-name=atr-v3-repr-compare
#SBATCH --partition=compute-short
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=results/slurm/visual_repr_compare_%j.out
#SBATCH --error=results/slurm/visual_repr_compare_%j.err

set -euo pipefail
ATR_REPRESENTATION_COMPARISON_CONFIG="${ATR_REPRESENTATION_COMPARISON_CONFIG:-configs/visual_representation_comparison_v1.json}"
ATR_REPRESENTATION_COMPARISON_OUTPUT="${ATR_REPRESENTATION_COMPARISON_OUTPUT:-results/final_visual_comparison}"
mkdir -p results/slurm "${ATR_REPRESENTATION_COMPARISON_OUTPUT}"
.venv/bin/python scripts/compare_visual_representations.py \
  --config "${ATR_REPRESENTATION_COMPARISON_CONFIG}" \
  --output "${ATR_REPRESENTATION_COMPARISON_OUTPUT}"

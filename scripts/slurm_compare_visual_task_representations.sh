#!/usr/bin/env bash
#SBATCH --job-name=atr-task-repr-compare
#SBATCH --partition=compute-short
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=results/slurm/task_repr_compare_%j.out
#SBATCH --error=results/slurm/task_repr_compare_%j.err

set -euo pipefail
: "${ATR_TASK_REPRESENTATION_COMPARISON_CONFIG:?set ATR_TASK_REPRESENTATION_COMPARISON_CONFIG}"
: "${ATR_TASK_REPRESENTATION_COMPARISON_OUTPUT:?set ATR_TASK_REPRESENTATION_COMPARISON_OUTPUT}"
mkdir -p results/slurm "${ATR_TASK_REPRESENTATION_COMPARISON_OUTPUT}"
.venv/bin/python scripts/compare_visual_task_representations.py \
  --config "${ATR_TASK_REPRESENTATION_COMPARISON_CONFIG}" \
  --output "${ATR_TASK_REPRESENTATION_COMPARISON_OUTPUT}"

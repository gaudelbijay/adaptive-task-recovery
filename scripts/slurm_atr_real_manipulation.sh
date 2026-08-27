#!/usr/bin/env bash
#SBATCH --job-name=atr-real-grasp
#SBATCH --partition=compute-v2
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=results/slurm/atr_real_grasp_%j.out
#SBATCH --error=results/slurm/atr_real_grasp_%j.err

set -euo pipefail

ATR_PYTHON="${ATR_PYTHON:-.venv/bin/python}"
"${ATR_PYTHON}" scripts/evaluate_atr_real_manipulation.py \
  --episodes "${ATR_REAL_GRASP_EPISODES:-10}"

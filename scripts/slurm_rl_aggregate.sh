#!/usr/bin/env bash
#SBATCH --job-name=atr-rl-agg
#SBATCH --partition=compute-v2
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#SBATCH --output=results/slurm/rl_agg_%j.out
#SBATCH --error=results/slurm/rl_agg_%j.err

set -euo pipefail

ATR_PYTHON="${ATR_PYTHON:-.venv/bin/python}"
"${ATR_PYTHON}" scripts/aggregate_rl_training.py

#!/usr/bin/env bash
#SBATCH --job-name=atr-bowl-sweep
#SBATCH --partition=compute-v2
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=results/slurm/bowl_sweep_%A_%a.out
#SBATCH --error=results/slurm/bowl_sweep_%A_%a.err

set -euo pipefail

ATR_PYTHON="${ATR_PYTHON:-.venv/bin/python}"
"${ATR_PYTHON}" scripts/diagnose_bowl_grasp.py --candidate "${SLURM_ARRAY_TASK_ID}"

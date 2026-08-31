#!/usr/bin/env bash
#SBATCH --job-name=atr-v4-temporal
#SBATCH --partition=gpu-l40s
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=results/slurm/v4_temporal_%j.out
#SBATCH --error=results/slurm/v4_temporal_%j.err

set -euo pipefail
mkdir -p results/slurm results/probes
export PYTHONUNBUFFERED=1
.venv/bin/python scripts/probe_v4_temporal_feasibility.py \
  --output results/probes/v4_temporal_feasibility_v4.json

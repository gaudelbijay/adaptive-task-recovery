#!/usr/bin/env bash
#SBATCH --job-name=atr-v58-stage
#SBATCH --partition=compute-short
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=results/slurm/v58_stage_%j.out
#SBATCH --error=results/slurm/v58_stage_%j.err
set -euo pipefail
mkdir -p results/slurm
.venv/bin/python scripts/build_v58_hierarchical_geometry_checkpoint.py --config configs/visual_recovery_v58_hierarchical_geometry_smoke.json --output "${ATR_VISUAL_OUTPUT:-results/visual_recovery_ppo}"

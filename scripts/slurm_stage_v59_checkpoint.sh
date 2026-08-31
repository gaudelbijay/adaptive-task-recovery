#!/usr/bin/env bash
#SBATCH --job-name=atr-v59-stage
#SBATCH --partition=compute-short
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=results/slurm/v59_stage_%j.out
#SBATCH --error=results/slurm/v59_stage_%j.err
set -euo pipefail; mkdir -p results/slurm
.venv/bin/python scripts/build_v58_hierarchical_geometry_checkpoint.py --config configs/visual_recovery_v59_renderer_v39_smoke.json --output "${ATR_VISUAL_OUTPUT:-results/visual_recovery_ppo}"

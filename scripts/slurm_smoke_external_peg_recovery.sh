#!/usr/bin/env bash
#SBATCH --job-name=atr-peg-recovery-smoke
#SBATCH --partition=gpu-l40s
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=results/slurm/peg_recovery_smoke_%j.out
#SBATCH --error=results/slurm/peg_recovery_smoke_%j.err

set -euo pipefail
mkdir -p results/slurm
export PYTHONUNBUFFERED=1
.venv/bin/python scripts/smoke_external_peg_recovery.py

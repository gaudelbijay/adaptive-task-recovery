#!/usr/bin/env bash
#SBATCH --job-name=atr-reboot-data
#SBATCH --partition=compute
#SBATCH --cpus-per-task=24
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=results/slurm/reboot_data_%j.out
#SBATCH --error=results/slurm/reboot_data_%j.err

set -euo pipefail
mkdir -p results/slurm results/reboot
export PYTHONUNBUFFERED=1
.venv/bin/python scripts/prepare_reboot_external_benchmark.py

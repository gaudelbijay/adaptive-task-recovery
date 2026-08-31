#!/usr/bin/env bash
#SBATCH --job-name=atr-v4-rev-xfer
#SBATCH --partition=gpu-l40s
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=results/slurm/v4_rev_xfer_%j.out
#SBATCH --error=results/slurm/v4_rev_xfer_%j.err

set -euo pipefail
mkdir -p results/slurm results/learned_recovery_v4
.venv/bin/python scripts/train_manipulation_ppo.py \
  --config configs/learned_recovery_v4_reverse_transfer_pilot.json \
  --output results/learned_recovery_v4

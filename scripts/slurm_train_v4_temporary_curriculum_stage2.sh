#!/usr/bin/env bash
#SBATCH --job-name=atr-v4-temp-cur2
#SBATCH --partition=gpu-l40s
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=results/slurm/v4_temp_cur2_%j.out
#SBATCH --error=results/slurm/v4_temp_cur2_%j.err

set -euo pipefail
mkdir -p results/slurm results/learned_recovery_v4
.venv/bin/python scripts/train_manipulation_ppo.py \
  --config configs/learned_recovery_v4_temporary_curriculum_stage2.json \
  --output results/learned_recovery_v4

#!/usr/bin/env bash
#SBATCH --job-name=atr-v4-contract
#SBATCH --partition=gpu-l40s
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=results/slurm/v4_teacher_contract_%j.out
#SBATCH --error=results/slurm/v4_teacher_contract_%j.err

set -euo pipefail
mkdir -p results/slurm
.venv/bin/python scripts/validate_v4_state_teacher_contract.py

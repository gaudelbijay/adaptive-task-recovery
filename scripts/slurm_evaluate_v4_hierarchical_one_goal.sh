#!/usr/bin/env bash
#SBATCH --job-name=atr-v4-hier-one
#SBATCH --partition=gpu-l40s
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=results/slurm/v4_hier_one_%j.out
#SBATCH --error=results/slurm/v4_hier_one_%j.err

set -euo pipefail
mkdir -p results/slurm
.venv/bin/python scripts/evaluate_v4_hierarchical_one_goal.py

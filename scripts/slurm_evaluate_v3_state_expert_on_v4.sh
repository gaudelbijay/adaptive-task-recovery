#!/usr/bin/env bash
#SBATCH --job-name=atr-v3-on-v4
#SBATCH --partition=gpu-l40s
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --array=0-2
#SBATCH --output=results/slurm/v3_on_v4_%A_%a.out
#SBATCH --error=results/slurm/v3_on_v4_%A_%a.err

set -euo pipefail
mkdir -p results/slurm results/v3_state_expert_on_v4
.venv/bin/python scripts/evaluate_v3_state_expert_on_v4.py \
  --task-index "${SLURM_ARRAY_TASK_ID}"

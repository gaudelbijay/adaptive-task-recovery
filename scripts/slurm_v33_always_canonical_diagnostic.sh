#!/usr/bin/env bash
#SBATCH --job-name=atr-v33-force-canonical
#SBATCH --partition=gpu-l40s
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=results/slurm/v33_force_canonical_%A_%a.out
#SBATCH --error=results/slurm/v33_force_canonical_%A_%a.err

set -euo pipefail
mkdir -p results/slurm
export PYTHONUNBUFFERED=1
conditions=(nominal intervention)
condition="${conditions[${SLURM_ARRAY_TASK_ID:-0}]}"
.venv/bin/python scripts/evaluate_v33_always_canonical_diagnostic.py \
  --config configs/visual_recovery_v19_canonical_view_v33_smoke.json \
  --output results/visual_recovery_ppo \
  --task-index 0 \
  --episodes 256 \
  --num-envs 32 \
  --seed-base 81000000 \
  --condition "${condition}" \
  --progress-head-mode normal \
  --visual-perturbation pixel_shift_right_4 \
  --environment-profile nominal

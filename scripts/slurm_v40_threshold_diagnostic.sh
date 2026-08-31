#!/usr/bin/env bash
#SBATCH --job-name=atr-v40-threshold-diagnostic
#SBATCH --partition=gpu-l40s
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --array=0-5%6
#SBATCH --output=results/slurm/v40_threshold_diagnostic_%A_%a.out
#SBATCH --error=results/slurm/v40_threshold_diagnostic_%A_%a.err
set -euo pipefail
thresholds=(0.005 0.010 0.015 0.020 0.030 0.040)
threshold="${thresholds[$SLURM_ARRAY_TASK_ID]}"; tag="${threshold/./p}"
export ATR_V40_MAGNITUDE_THRESHOLD="$threshold" PYTHONUNBUFFERED=1
.venv/bin/python scripts/evaluate_v40_threshold_diagnostic.py \
  --config configs/visual_recovery_v19_backkey_v40_smoke.json \
  --output "results/diagnostics/v40_threshold_${tag}" --task-index 0 \
  --episodes 128 --num-envs 32 --seed-base 123000000 --condition nominal \
  --progress-head-mode normal --visual-perturbation none --environment-profile nominal

#!/usr/bin/env bash
#SBATCH --job-name=atr-v40-threshold015-domains
#SBATCH --partition=gpu-l40s
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --array=0-4%5
#SBATCH --output=results/slurm/v40_threshold015_domains_%A_%a.out
#SBATCH --error=results/slurm/v40_threshold015_domains_%A_%a.err
set -euo pipefail
case "$SLURM_ARRAY_TASK_ID" in
  0) condition=nominal; perturbation=none; profile=lighting_back_key ;;
  1) condition=intervention; perturbation=none; profile=lighting_back_key ;;
  2) condition=intervention; perturbation=subpixel_shift_left_1_5; profile=nominal ;;
  3) condition=intervention; perturbation=scale_95; profile=nominal ;;
  4) condition=intervention; perturbation=none; profile=camera_back_3cm ;;
esac
export ATR_V40_MAGNITUDE_THRESHOLD=0.015 PYTHONUNBUFFERED=1
.venv/bin/python scripts/evaluate_v40_threshold_diagnostic.py \
  --config configs/visual_recovery_v19_backkey_v40_smoke.json \
  --output results/diagnostics/v40_threshold_0p015 --task-index 0 \
  --episodes 128 --num-envs 32 --seed-base 123000000 --condition "$condition" \
  --progress-head-mode normal --visual-perturbation "$perturbation" --environment-profile "$profile"

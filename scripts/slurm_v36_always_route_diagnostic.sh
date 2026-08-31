#!/usr/bin/env bash
#SBATCH --job-name=atr-v36-route-diagnostic
#SBATCH --partition=gpu-l40s
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --array=0-4%5
#SBATCH --output=results/slurm/v36_always_route_diagnostic_%A_%a.out
#SBATCH --error=results/slurm/v36_always_route_diagnostic_%A_%a.err

set -euo pipefail
mkdir -p results/slurm
export PYTHONUNBUFFERED=1

case "${SLURM_ARRAY_TASK_ID}" in
  0) perturbation=none; profile=nominal ;;
  1) perturbation=subpixel_shift_left_1_5; profile=nominal ;;
  2) perturbation=scale_95; profile=nominal ;;
  3) perturbation=none; profile=camera_back_3cm ;;
  4) perturbation=none; profile=lighting_back_key ;;
  *) exit 2 ;;
esac

.venv/bin/python scripts/evaluate_v36_always_route_diagnostic.py \
  --config configs/visual_recovery_v19_continuous_canonical_v36_smoke.json \
  --output results/diagnostics/v36_always_route \
  --task-index 0 --episodes 128 --num-envs 32 --seed-base 119000000 \
  --condition intervention --progress-head-mode normal \
  --visual-perturbation "${perturbation}" --environment-profile "${profile}"

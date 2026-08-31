#!/usr/bin/env bash
#SBATCH --job-name=atr-v38-magnitude-diagnostic
#SBATCH --partition=gpu-l40s
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --array=0-7%8
#SBATCH --output=results/slurm/v38_magnitude_diagnostic_%A_%a.out
#SBATCH --error=results/slurm/v38_magnitude_diagnostic_%A_%a.err

set -euo pipefail
mkdir -p results/slurm
thresholds=(0.003 0.005 0.007 0.010)
if (( SLURM_ARRAY_TASK_ID < 4 )); then
  threshold="${thresholds[$SLURM_ARRAY_TASK_ID]}"; perturbation=none; profile=nominal; condition=nominal
else
  threshold=0.005
  case "$SLURM_ARRAY_TASK_ID" in
    4) perturbation=subpixel_shift_left_1_5; profile=nominal; condition=intervention ;;
    5) perturbation=scale_95; profile=nominal; condition=intervention ;;
    6) perturbation=none; profile=lighting_back_key; condition=nominal ;;
    7) perturbation=none; profile=camera_back_3cm; condition=intervention ;;
  esac
fi
tag="${threshold/./p}"
export ATR_V38_MAGNITUDE_THRESHOLD="$threshold" PYTHONUNBUFFERED=1
.venv/bin/python scripts/evaluate_v38_magnitude_gate_diagnostic.py \
  --config configs/visual_recovery_v19_cardinality_aligned_v38_smoke.json \
  --output "results/diagnostics/v38_magnitude_${tag}" --task-index 0 \
  --episodes 128 --num-envs 32 --seed-base 121000000 --condition "$condition" \
  --progress-head-mode normal --visual-perturbation "$perturbation" --environment-profile "$profile"

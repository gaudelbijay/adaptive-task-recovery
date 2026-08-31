#!/usr/bin/env bash
#SBATCH --job-name=atr-v4-controller
#SBATCH --partition=gpu-l40s
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --array=0-14
#SBATCH --output=results/slurm/v4_controller_v27_%A_%a.out
#SBATCH --error=results/slurm/v4_controller_v27_%A_%a.err

set -euo pipefail
ATR_V4_CONTROLLER_OUTPUT="${ATR_V4_CONTROLLER_OUTPUT:-results/v4_temporal_controller_v27_ood}"
mkdir -p results/slurm "${ATR_V4_CONTROLLER_OUTPUT}"
export PYTHONUNBUFFERED=1
OOD_ARGS=()
if [[ "${ATR_V4_PROFILE:-nominal}" != "nominal" ]]; then
  OOD_ARGS+=(--env-id LearnedRecovery-v4-OOD --visual-domain-profile "${ATR_V4_PROFILE}")
fi
.venv/bin/python scripts/evaluate_v4_temporal_controller.py \
  --task-index "${SLURM_ARRAY_TASK_ID}" \
  --classifier results/models/v4_temporal_dinov2_productive_wait_v9.joblib \
  --classifier-metadata results/models/v4_temporal_dinov2_productive_wait_v9.json \
  --output-dir "${ATR_V4_CONTROLLER_OUTPUT}" \
  --steps "${ATR_V4_STEPS:-240}" \
  "${OOD_ARGS[@]}" \
  --seed-base "${ATR_V4_SEED_BASE:-180000000}"

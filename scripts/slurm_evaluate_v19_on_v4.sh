#!/usr/bin/env bash
#SBATCH --job-name=atr-v19-v4
#SBATCH --partition=gpu-l40s
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --array=0-44
#SBATCH --output=results/slurm/v19_v4_%A_%a.out
#SBATCH --error=results/slurm/v19_v4_%A_%a.err

set -euo pipefail
ATR_V4_OUTPUT_DIR="${ATR_V4_OUTPUT_DIR:-results/v19_on_v4_oracle_wait_from_onset}"
mkdir -p results/slurm "${ATR_V4_OUTPUT_DIR}"
export PYTHONUNBUFFERED=1
OOD_ARGS=()
if [[ "${ATR_V4_PROFILE:-nominal}" != "nominal" ]]; then
  OOD_ARGS+=(--env-id LearnedRecovery-v4-OOD --visual-domain-profile "${ATR_V4_PROFILE}")
fi
.venv/bin/python scripts/evaluate_v19_on_v4.py \
  --config "${ATR_V4_CONFIG:-configs/visual_recovery_dual_specialist_dagger_v19.json}" \
  --checkpoint-root "${ATR_V4_CHECKPOINT_ROOT:-results/visual_recovery_ppo/visual_recovery_dual_specialist_dagger_v19}" \
  --task-index "${SLURM_ARRAY_TASK_ID}" \
  --steps "${ATR_V4_STEPS:-240}" \
  --seed-base "${ATR_V4_SEED_BASE:-180000000}" \
  "${OOD_ARGS[@]}" \
  --output-dir "${ATR_V4_OUTPUT_DIR}"

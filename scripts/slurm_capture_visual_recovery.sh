#!/usr/bin/env bash
#SBATCH --job-name=atr-visual-video
#SBATCH --partition=gpu-l40s
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=results/slurm/visual_video_%A_%a.out
#SBATCH --error=results/slurm/visual_video_%A_%a.err

set -euo pipefail

ATR_NODE_CACHE="${SLURM_TMPDIR:-/tmp}/atr-visual-video-${SLURM_JOB_ID}-${SLURM_ARRAY_TASK_ID}"
mkdir -p "${ATR_NODE_CACHE}/gl" "${ATR_NODE_CACHE}/cuda" "${ATR_NODE_CACHE}/xdg" results/slurm
export __GL_SHADER_DISK_CACHE_PATH="${ATR_NODE_CACHE}/gl"
export CUDA_CACHE_PATH="${ATR_NODE_CACHE}/cuda"
export XDG_CACHE_HOME="${ATR_NODE_CACHE}/xdg"
export PYTHONUNBUFFERED=1

case "${SLURM_ARRAY_TASK_ID}" in
  0) ATR_BRANCH=first_goal_removed ;;
  1) ATR_BRANCH=second_goal_removed ;;
  2) ATR_BRANCH=nominal ;;
  *) echo "invalid array index: ${SLURM_ARRAY_TASK_ID}" >&2; exit 2 ;;
esac

: "${ATR_VISUAL_CAPTURE_CONFIG:?set ATR_VISUAL_CAPTURE_CONFIG to the validated winner}"
capture_args=(
  --config "${ATR_VISUAL_CAPTURE_CONFIG}"
  --results "${ATR_VISUAL_CAPTURE_RESULTS:-results/visual_recovery_ppo}"
  --output "${ATR_VISUAL_CAPTURE_OUTPUT:-results/visual_recovery_ppo/videos}"
  --task-index "${ATR_VISUAL_CAPTURE_TASK_INDEX:-0}"
  --branch "${ATR_BRANCH}"
)
if [[ -n "${ATR_VISUAL_CAPTURE_SELECTION_ARTIFACT:-}" ]]; then
  capture_args+=(
    --selection-artifact "${ATR_VISUAL_CAPTURE_SELECTION_ARTIFACT}"
    --expected-selection "${ATR_VISUAL_CAPTURE_EXPECTED_SELECTION:?set expected selection}"
  )
elif [[ -n "${ATR_VISUAL_CAPTURE_AGGREGATE:-}" ]]; then
  capture_args+=(--aggregate "${ATR_VISUAL_CAPTURE_AGGREGATE}")
fi
if [[ -n "${ATR_VISUAL_CAPTURE_STRICT_CONFIG:-}" && "${ATR_BRANCH}" != nominal ]]; then
  capture_args+=(--strict-config "${ATR_VISUAL_CAPTURE_STRICT_CONFIG}")
fi
.venv/bin/python scripts/capture_visual_recovery_policy.py "${capture_args[@]}"

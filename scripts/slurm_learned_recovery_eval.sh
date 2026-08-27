#!/usr/bin/env bash
#SBATCH --job-name=atr-recovery-eval
#SBATCH --partition=gpu-l40s
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=results/slurm/recovery_eval_%A_%a.out
#SBATCH --error=results/slurm/recovery_eval_%A_%a.err

set -euo pipefail

ATR_NODE_CACHE="${SLURM_TMPDIR:-/tmp}/atr-recovery-eval-${SLURM_JOB_ID}-${SLURM_ARRAY_TASK_ID}"
mkdir -p "${ATR_NODE_CACHE}/gl" "${ATR_NODE_CACHE}/cuda" "${ATR_NODE_CACHE}/xdg"
export __GL_SHADER_DISK_CACHE_PATH="${ATR_NODE_CACHE}/gl"
export CUDA_CACHE_PATH="${ATR_NODE_CACHE}/cuda"
export XDG_CACHE_HOME="${ATR_NODE_CACHE}/xdg"
export PYTHONUNBUFFERED=1

ATR_RECOVERY_CONFIG="${ATR_RECOVERY_CONFIG:-configs/learned_recovery_ppo_v1.json}"
ATR_RECOVERY_OUTPUT="${ATR_RECOVERY_OUTPUT:-results/learned_recovery}"
ATR_PYTHON="${ATR_PYTHON:-.venv/bin/python}"

for ATR_CONDITION in intervention nominal; do
  "${ATR_PYTHON}" scripts/evaluate_manipulation_ppo.py \
    --config "${ATR_RECOVERY_CONFIG}" \
    --output "${ATR_RECOVERY_OUTPUT}" \
    --task-index "${SLURM_ARRAY_TASK_ID}" \
    --episodes "${ATR_EVAL_EPISODES:-256}" \
    --num-envs "${ATR_EVAL_NUM_ENVS:-32}" \
    --condition "${ATR_CONDITION}"
done

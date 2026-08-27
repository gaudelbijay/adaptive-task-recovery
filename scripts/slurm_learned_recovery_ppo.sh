#!/usr/bin/env bash
#SBATCH --job-name=atr-learned-recovery
#SBATCH --partition=gpu-l40s
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --signal=B:USR1@300
#SBATCH --output=results/slurm/recovery_%A_%a.out
#SBATCH --error=results/slurm/recovery_%A_%a.err

set -euo pipefail

ATR_NODE_CACHE="${SLURM_TMPDIR:-/tmp}/atr-recovery-${SLURM_JOB_ID}-${SLURM_ARRAY_TASK_ID}"
mkdir -p "${ATR_NODE_CACHE}/gl" "${ATR_NODE_CACHE}/cuda" "${ATR_NODE_CACHE}/xdg"
export __GL_SHADER_DISK_CACHE_PATH="${ATR_NODE_CACHE}/gl"
export CUDA_CACHE_PATH="${ATR_NODE_CACHE}/cuda"
export XDG_CACHE_HOME="${ATR_NODE_CACHE}/xdg"
export PYTHONUNBUFFERED=1

ATR_RECOVERY_CONFIG="${ATR_RECOVERY_CONFIG:-configs/learned_recovery_ppo_v1.json}"
ATR_RECOVERY_OUTPUT="${ATR_RECOVERY_OUTPUT:-results/learned_recovery}"
ATR_PYTHON="${ATR_PYTHON:-.venv/bin/python}"

"${ATR_PYTHON}" scripts/train_manipulation_ppo.py \
  --config "${ATR_RECOVERY_CONFIG}" \
  --output "${ATR_RECOVERY_OUTPUT}" \
  --task-index "${SLURM_ARRAY_TASK_ID}"

ATR_TASK_STDERR="results/slurm/recovery_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.err"
if grep -qi "buffer overflow detected" "${ATR_TASK_STDERR}"; then
  echo "fatal: simulator buffer overflow detected; quarantine this checkpoint" >&2
  exit 42
fi

# Jarvis limits a job to 24 hours. SIGUSR1 makes the trainer atomically save
# at an update boundary; an incomplete task then resubmits only itself and the
# next allocation resumes model, optimizer, counters, and RNG from latest.pt.
if ! "${ATR_PYTHON}" scripts/check_manipulation_training_complete.py \
  --config "${ATR_RECOVERY_CONFIG}" \
  --output "${ATR_RECOVERY_OUTPUT}" \
  --task-index "${SLURM_ARRAY_TASK_ID}"; then
  sbatch --array="${SLURM_ARRAY_TASK_ID}" \
    --export="ALL,ATR_RECOVERY_CONFIG=${ATR_RECOVERY_CONFIG},ATR_RECOVERY_OUTPUT=${ATR_RECOVERY_OUTPUT},ATR_PYTHON=${ATR_PYTHON}" \
    scripts/slurm_learned_recovery_ppo.sh
fi

#!/usr/bin/env bash
# Submit as an array. The same element requeues and resumes exact RNG state.
# SIGUSR1 asks Python to finish its current short episode and save atomically.
#SBATCH --job-name=atr-rl-train
#SBATCH --partition=compute-v2
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=24:00:00
#SBATCH --signal=USR1@180
#SBATCH --requeue
#SBATCH --output=results/slurm/rl_%A_%a.out
#SBATCH --error=results/slurm/rl_%A_%a.err

set -euo pipefail

ATR_RL_CONFIG="${ATR_RL_CONFIG:-configs/rl_training_v1.json}"
ATR_RL_OUTPUT="${ATR_RL_OUTPUT:-results/rl_training}"
ATR_PYTHON="${ATR_PYTHON:-.venv/bin/python}"

test -n "${SLURM_ARRAY_TASK_ID:-}" || {
  echo "SLURM_ARRAY_TASK_ID is missing; submit this script as an array" >&2
  exit 2
}

set +e
"${ATR_PYTHON}" scripts/train_rl_policy.py \
  --config "${ATR_RL_CONFIG}" \
  --output "${ATR_RL_OUTPUT}" \
  --task-index "${SLURM_ARRAY_TASK_ID}"
ATR_TRAIN_EXIT=$?
set -e

if [[ "${ATR_TRAIN_EXIT}" == "75" ]]; then
  ATR_REQUEUE_ID="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID}}_${SLURM_ARRAY_TASK_ID}"
  scontrol requeue "${ATR_REQUEUE_ID}"
elif [[ "${ATR_TRAIN_EXIT}" != "0" ]]; then
  exit "${ATR_TRAIN_EXIT}"
fi

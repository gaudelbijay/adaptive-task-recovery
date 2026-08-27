#!/usr/bin/env bash
# Submit as an array. Re-submit the identical array to resume exact RNG state.
# The USR1 warning gives Python time to finish its current short episode; the
# latest periodic checkpoint bounds lost work even if Slurm terminates it.
#SBATCH --job-name=atr-rl-train
#SBATCH --partition=compute-v2
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=24:00:00
#SBATCH --signal=B:USR1@180
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

"${ATR_PYTHON}" scripts/train_rl_policy.py \
  --config "${ATR_RL_CONFIG}" \
  --output "${ATR_RL_OUTPUT}" \
  --task-index "${SLURM_ARRAY_TASK_ID}"

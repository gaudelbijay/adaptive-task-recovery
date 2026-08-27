#!/usr/bin/env bash
#SBATCH --job-name=atr-manip-ppo
#SBATCH --partition=gpu-l40s
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --signal=B:USR1@300
#SBATCH --output=results/slurm/manip_%A_%a.out
#SBATCH --error=results/slurm/manip_%A_%a.err

set -euo pipefail

# NVIDIA/SAPIEN otherwise place shader caches under ~/.cache on Jarvis's
# shared MMFS.  That produced prolonged uninterruptible I/O wait and 0% GPU
# utilization in the first pilot.  Keep ephemeral caches on node-local disk.
ATR_NODE_CACHE="${SLURM_TMPDIR:-/tmp}/atr-${SLURM_JOB_ID}-${SLURM_ARRAY_TASK_ID}"
mkdir -p "${ATR_NODE_CACHE}/gl" "${ATR_NODE_CACHE}/cuda" "${ATR_NODE_CACHE}/xdg"
export __GL_SHADER_DISK_CACHE_PATH="${ATR_NODE_CACHE}/gl"
export CUDA_CACHE_PATH="${ATR_NODE_CACHE}/cuda"
export XDG_CACHE_HOME="${ATR_NODE_CACHE}/xdg"
export PYTHONUNBUFFERED=1

ATR_MANIP_CONFIG="${ATR_MANIP_CONFIG:-configs/manipulation_ppo_v1.json}"
ATR_MANIP_OUTPUT="${ATR_MANIP_OUTPUT:-results/manipulation_ppo}"
ATR_PYTHON="${ATR_PYTHON:-.venv/bin/python}"

"${ATR_PYTHON}" scripts/train_manipulation_ppo.py \
  --config "${ATR_MANIP_CONFIG}" \
  --output "${ATR_MANIP_OUTPUT}" \
  --task-index "${SLURM_ARRAY_TASK_ID}"

# Native PhysX capacity failures are logged but do not necessarily terminate
# Python. Convert them into a failed Slurm task so held-out evaluation cannot
# consume a silently corrupted run.
ATR_TASK_STDERR="results/slurm/manip_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.err"
if grep -qi "buffer overflow detected" "${ATR_TASK_STDERR}"; then
  echo "fatal: simulator buffer overflow detected; quarantine this checkpoint" >&2
  exit 42
fi

#!/usr/bin/env bash
#SBATCH --job-name=atr-ne-dreamer
#SBATCH --partition=gpu-l40s
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --signal=USR1@300
#SBATCH --requeue
#SBATCH --output=results/slurm/ne_dreamer_%A_%a.out
#SBATCH --error=results/slurm/ne_dreamer_%A_%a.err

set -euo pipefail

ATR_NE_CONFIG="${ATR_NE_CONFIG:-configs/learned_recovery_nedreamer_pilot.json}"
ATR_NE_OUTPUT="${ATR_NE_OUTPUT:-results/vision_nedreamer}"
ATR_NE_UPSTREAM="${ATR_NE_UPSTREAM:-/home/bgaudel/.cache/atr/nedreamer-7ca2193}"
ATR_PYTHON="${ATR_PYTHON:-.venv/bin/python}"
ATR_TASK_INDEX="${SLURM_ARRAY_TASK_ID:-0}"
ATR_NODE_CACHE="${SLURM_TMPDIR:-/tmp}/atr-ne-${SLURM_JOB_ID}-${ATR_TASK_INDEX}"
mkdir -p "${ATR_NODE_CACHE}/gl" "${ATR_NODE_CACHE}/cuda" "${ATR_NODE_CACHE}/xdg" results/slurm
export __GL_SHADER_DISK_CACHE_PATH="${ATR_NODE_CACHE}/gl"
export CUDA_CACHE_PATH="${ATR_NODE_CACHE}/cuda"
export XDG_CACHE_HOME="${ATR_NODE_CACHE}/xdg"
export PYTHONUNBUFFERED=1
export WANDB_MODE=disabled
export WANDB_SILENT=true

"${ATR_PYTHON}" scripts/train_vision_nedreamer.py \
  --config "${ATR_NE_CONFIG}" \
  --output "${ATR_NE_OUTPUT}" \
  --task-index "${ATR_TASK_INDEX}" \
  --upstream "${ATR_NE_UPSTREAM}"

ATR_COMPLETE=$("${ATR_PYTHON}" -c 'import json, pathlib, sys; c=json.load(open(sys.argv[1])); s=c["seeds"][int(sys.argv[3])]; p=pathlib.Path(sys.argv[2])/c["name"]/c["algorithm"]/("seed_"+str(s))/"TRAINING_COMPLETE.json"; print(int(p.exists()))' "${ATR_NE_CONFIG}" "${ATR_NE_OUTPUT}" "${ATR_TASK_INDEX}")
if [[ "${ATR_COMPLETE}" != "1" ]]; then
  ATR_REQUEUE_ID="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID}}_${ATR_TASK_INDEX}"
  scontrol requeue "${ATR_REQUEUE_ID}"
fi

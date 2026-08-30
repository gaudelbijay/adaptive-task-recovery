#!/usr/bin/env bash
#SBATCH --job-name=atr-visual-ppo
#SBATCH --partition=gpu-l40s
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=24:00:00
#SBATCH --signal=USR1@300
#SBATCH --requeue
#SBATCH --output=results/slurm/visual_ppo_%A_%a.out
#SBATCH --error=results/slurm/visual_ppo_%A_%a.err

set -euo pipefail

ATR_VISUAL_CONFIG="${ATR_VISUAL_CONFIG:-configs/visual_recovery_ppo_gate_v1.json}"
ATR_VISUAL_OUTPUT="${ATR_VISUAL_OUTPUT:-results/visual_recovery_ppo}"
ATR_PYTHON="${ATR_PYTHON:-.venv/bin/python}"
ATR_TASK_INDEX="${SLURM_ARRAY_TASK_ID:-0}"
ATR_NODE_CACHE="${SLURM_TMPDIR:-/tmp}/atr-visual-${SLURM_JOB_ID}-${ATR_TASK_INDEX}"
mkdir -p "${ATR_NODE_CACHE}/gl" "${ATR_NODE_CACHE}/cuda" "${ATR_NODE_CACHE}/xdg" results/slurm
export __GL_SHADER_DISK_CACHE_PATH="${ATR_NODE_CACHE}/gl"
export CUDA_CACHE_PATH="${ATR_NODE_CACHE}/cuda"
export XDG_CACHE_HOME="${ATR_NODE_CACHE}/xdg"
export PYTHONUNBUFFERED=1

"${ATR_PYTHON}" scripts/train_visual_recovery_ppo.py \
  --config "${ATR_VISUAL_CONFIG}" \
  --output "${ATR_VISUAL_OUTPUT}" \
  --task-index "${ATR_TASK_INDEX}"

ATR_COMPLETE=$("${ATR_PYTHON}" -c 'import json,pathlib,sys; c=json.load(open(sys.argv[1])); tasks=[(e,s) for e in c["experiments"] for s in c["seeds"]]; e,s=tasks[int(sys.argv[3])]; p=pathlib.Path(sys.argv[2])/c["name"]/e["method"]/("seed_"+str(s))/"TRAINING_COMPLETE.json"; print(int(p.exists()))' "${ATR_VISUAL_CONFIG}" "${ATR_VISUAL_OUTPUT}" "${ATR_TASK_INDEX}")
if [[ "${ATR_COMPLETE}" != "1" ]]; then
  # Requeue this array element instead of submitting a successor job.  Keeping
  # the same Slurm identity prevents afterok evaluation jobs from being
  # released between 24-hour allocations.  The trainer resumes exactly from
  # latest.pt, including optimizer, counters, and RNG state.
  ATR_REQUEUE_ID="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID}}_${ATR_TASK_INDEX}"
  scontrol requeue "${ATR_REQUEUE_ID}"
fi

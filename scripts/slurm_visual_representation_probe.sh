#!/usr/bin/env bash
#SBATCH --job-name=atr-visual-probe
#SBATCH --partition=gpu-l40s
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=results/slurm/visual_probe_%A_%a.out
#SBATCH --error=results/slurm/visual_probe_%A_%a.err

set -euo pipefail
ATR_VISUAL_CONFIG="${ATR_VISUAL_CONFIG:-configs/visual_recovery_intervention_v1.json}"
ATR_VISUAL_OUTPUT="${ATR_VISUAL_OUTPUT:-results/visual_recovery_ppo}"
ATR_PYTHON="${ATR_PYTHON:-.venv/bin/python}"
ATR_TASK_INDEX="${SLURM_ARRAY_TASK_ID:-0}"
ATR_NODE_CACHE="${SLURM_TMPDIR:-/tmp}/atr-probe-${SLURM_JOB_ID}-${ATR_TASK_INDEX}"
mkdir -p "${ATR_NODE_CACHE}/gl" "${ATR_NODE_CACHE}/cuda" "${ATR_NODE_CACHE}/xdg" results/slurm
export __GL_SHADER_DISK_CACHE_PATH="${ATR_NODE_CACHE}/gl"
export CUDA_CACHE_PATH="${ATR_NODE_CACHE}/cuda"
export XDG_CACHE_HOME="${ATR_NODE_CACHE}/xdg"
export PYTHONUNBUFFERED=1
probe_args=(
  --config "${ATR_VISUAL_CONFIG}" --output "${ATR_VISUAL_OUTPUT}" \
  --task-index "${ATR_TASK_INDEX}" --samples "${ATR_PROBE_SAMPLES:-8192}" \
  --num-envs "${ATR_PROBE_NUM_ENVS:-32}" --ridge "${ATR_PROBE_RIDGE:-1.0}" \
  --filename "${ATR_REPRESENTATION_PROBE_FILENAME:-representation_probe.json}"
)
if [[ -n "${ATR_PROBE_PROTOCOL_CONFIG:-}" ]]; then
  probe_args+=(--probe-protocol-config "${ATR_PROBE_PROTOCOL_CONFIG}")
fi
"${ATR_PYTHON}" scripts/probe_visual_representation.py "${probe_args[@]}"

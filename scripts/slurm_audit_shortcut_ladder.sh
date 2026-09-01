#!/usr/bin/env bash
#SBATCH --job-name=atr-ladder
#SBATCH --partition=gpu-l40s
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --output=results/slurm/shortcut_ladder_%j.out
#SBATCH --error=results/slurm/shortcut_ladder_%j.err

set -euo pipefail
mkdir -p results/slurm results/router/ladder
export PYTHONUNBUFFERED=1
EXTRA=()
if [[ -n "${ATR_LADDER_GEOMETRY_DIM:-}" ]]; then
  EXTRA+=(--geometry-dim "${ATR_LADDER_GEOMETRY_DIM}")
fi
if [[ -n "${ATR_LADDER_HELDOUT_OPTION:-}" ]]; then
  EXTRA+=(--heldout-option "${ATR_LADDER_HELDOUT_OPTION}")
fi
if [[ "${ATR_LADDER_PHYSICAL_HELDOUT_ONLY:-0}" == "1" ]]; then
  EXTRA+=(--physical-heldout-only)
fi
.venv/bin/python scripts/audit_shortcut_ladder.py \
  --data "${ATR_LADDER_DATA:?set ATR_LADDER_DATA}" \
  --metadata "${ATR_LADDER_METADATA:?set ATR_LADDER_METADATA}" \
  --checkpoint-dir "${ATR_LADDER_CHECKPOINTS:?set ATR_LADDER_CHECKPOINTS}" \
  --heuristic "${ATR_LADDER_HEURISTIC:-actor_pair}" \
  --output "${ATR_LADDER_OUTPUT:?set ATR_LADDER_OUTPUT}" \
  "${EXTRA[@]}"

#!/usr/bin/env bash
#SBATCH --job-name=atr-state-fallback-router
#SBATCH --partition=compute-short
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --output=results/slurm/state_fallback_router_%j.out
#SBATCH --error=results/slurm/state_fallback_router_%j.err

set -euo pipefail
mkdir -p results/slurm results/gates
ATR_STATE_FALLBACK_RELEASE_CONFIG="${ATR_STATE_FALLBACK_RELEASE_CONFIG:-configs/state_fallback_release_gate_v1.json}"
ATR_STATE_FALLBACK_RELEASE_OUTPUT="${ATR_STATE_FALLBACK_RELEASE_OUTPUT:-results/gates/state_fallback_release_v1.json}"
.venv/bin/python scripts/check_state_fallback_release_gate.py \
  --config "${ATR_STATE_FALLBACK_RELEASE_CONFIG}" \
  --output "${ATR_STATE_FALLBACK_RELEASE_OUTPUT}"

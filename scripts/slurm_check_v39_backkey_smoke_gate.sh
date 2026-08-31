#!/usr/bin/env bash
#SBATCH --job-name=atr-v39-smoke-gate
#SBATCH --partition=compute-short
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=results/slurm/v39_smoke_gate_%j.out
#SBATCH --error=results/slurm/v39_smoke_gate_%j.err

set -euo pipefail
: "${ATR_V39_GATE_CONFIG:?set ATR_V39_GATE_CONFIG}"
mkdir -p results/slurm results/gates
.venv/bin/python scripts/check_v36_continuous_canonical_smoke_gate.py \
  --config "${ATR_V39_GATE_CONFIG}" \
  --output "${ATR_V39_GATE_OUTPUT:-results/gates/v39_backkey_smoke_gate_v1.json}"

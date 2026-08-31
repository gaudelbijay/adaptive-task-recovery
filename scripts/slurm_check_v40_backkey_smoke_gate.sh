#!/usr/bin/env bash
#SBATCH --job-name=atr-v40-smoke-gate
#SBATCH --partition=compute-short
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=results/slurm/v40_smoke_gate_%j.out
#SBATCH --error=results/slurm/v40_smoke_gate_%j.err
set -euo pipefail
: "${ATR_V40_GATE_CONFIG:?set ATR_V40_GATE_CONFIG}"
mkdir -p results/slurm results/gates
.venv/bin/python scripts/check_v36_continuous_canonical_smoke_gate.py --config "$ATR_V40_GATE_CONFIG" --output "${ATR_V40_GATE_OUTPUT:-results/gates/v40_backkey_smoke_gate_v1.json}"

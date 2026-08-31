#!/usr/bin/env bash
#SBATCH --job-name=atr-v41-final-gate
#SBATCH --partition=compute-short
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=results/slurm/v41_final_gate_%j.out
#SBATCH --error=results/slurm/v41_final_gate_%j.err

set -euo pipefail
: "${ATR_V41_FINAL_GATE_CONFIG:?set ATR_V41_FINAL_GATE_CONFIG}"
mkdir -p results/slurm results/gates
.venv/bin/python scripts/check_v28_final_release_gate.py \
  --config "${ATR_V41_FINAL_GATE_CONFIG}" \
  --output "${ATR_V41_FINAL_GATE_OUTPUT:-results/gates/v41_final_evidence_gate_v1.json}"

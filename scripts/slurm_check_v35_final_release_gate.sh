#!/usr/bin/env bash
#SBATCH --job-name=atr-v35-final-gate
#SBATCH --partition=compute-short
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=results/slurm/v35_final_gate_%j.out
#SBATCH --error=results/slurm/v35_final_gate_%j.err

set -euo pipefail
: "${ATR_V35_FINAL_GATE_CONFIG:?set ATR_V35_FINAL_GATE_CONFIG}"
mkdir -p results/slurm results/gates
.venv/bin/python scripts/check_v35_final_release_gate.py \
  --config "${ATR_V35_FINAL_GATE_CONFIG}" \
  --output "${ATR_V35_FINAL_GATE_OUTPUT:-results/gates/v35_final_release_gate_v1.json}"

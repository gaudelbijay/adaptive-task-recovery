#!/usr/bin/env bash
#SBATCH --job-name=atr-integrated-5seed-gate
#SBATCH --partition=compute-short
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --output=results/slurm/integrated_5seed_gate_%j.out
#SBATCH --error=results/slurm/integrated_5seed_gate_%j.err

set -euo pipefail
: "${ATR_CONFIRM_CONFIG:?set ATR_CONFIRM_CONFIG}"
: "${ATR_CONFIRM_GATE_OUTPUT:?set ATR_CONFIRM_GATE_OUTPUT}"
.venv/bin/python scripts/check_integrated_five_seed_gate.py \
  --config "${ATR_CONFIRM_CONFIG}" --output "${ATR_CONFIRM_GATE_OUTPUT}"

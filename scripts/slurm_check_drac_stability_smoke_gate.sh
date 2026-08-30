#!/usr/bin/env bash
#SBATCH --job-name=atr-drac-smoke-gate
#SBATCH --partition=compute-short
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=results/slurm/drac_smoke_gate_%j.out
#SBATCH --error=results/slurm/drac_smoke_gate_%j.err

set -euo pipefail

ATR_DRAC_GATE_CONFIG="${ATR_DRAC_GATE_CONFIG:-configs/drac_stability_smoke_gate_v1.json}"
ATR_DRAC_GATE_OUTPUT="${ATR_DRAC_GATE_OUTPUT:-results/gates/drac_stability_smoke_gate_v1.json}"
mkdir -p results/slurm
.venv/bin/python scripts/check_drac_stability_smoke_gate.py \
  --config "${ATR_DRAC_GATE_CONFIG}" --output "${ATR_DRAC_GATE_OUTPUT}"

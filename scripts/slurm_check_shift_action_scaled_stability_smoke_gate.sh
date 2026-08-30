#!/usr/bin/env bash
#SBATCH --job-name=atr-shift-scaled-gate
#SBATCH --partition=compute-short
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=results/slurm/shift_scaled_gate_%j.out
#SBATCH --error=results/slurm/shift_scaled_gate_%j.err

set -euo pipefail

mkdir -p results/slurm results/gates
.venv/bin/python scripts/check_shift_action_scaled_stability_smoke_gate.py \
  --config configs/shift_action_scaled_stability_smoke_gate_v1.json \
  --output results/gates/shift_action_scaled_stability_smoke_gate_v1.json

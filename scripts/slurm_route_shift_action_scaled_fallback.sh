#!/usr/bin/env bash
#SBATCH --job-name=atr-shift-scaled-route
#SBATCH --partition=compute-short
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:05:00
#SBATCH --output=results/slurm/shift_scaled_route_%j.out
#SBATCH --error=results/slurm/shift_scaled_route_%j.err

set -euo pipefail

mkdir -p results/slurm results/gates
.venv/bin/python scripts/route_shift_action_scaled_fallback.py \
  --upstream-result results/gates/shift_action_stability_smoke_gate_v1.json \
  --gate-config configs/shift_action_stability_smoke_gate_v1.json \
  --output results/gates/shift_action_scaled_fallback_route_v1.json

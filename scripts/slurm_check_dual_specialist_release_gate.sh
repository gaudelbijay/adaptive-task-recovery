#!/usr/bin/env bash
#SBATCH --job-name=atr-dual-release-gate
#SBATCH --partition=compute-short
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --output=results/slurm/dual_release_gate_%j.out
#SBATCH --error=results/slurm/dual_release_gate_%j.err

set -euo pipefail
mkdir -p results/slurm results/gates
ATR_DUAL_RELEASE_CONFIG="${ATR_DUAL_RELEASE_CONFIG:-configs/dual_specialist_release_gate_v1.json}"
ATR_DUAL_RELEASE_OUTPUT="${ATR_DUAL_RELEASE_OUTPUT:-results/gates/dual_specialist_release_v1.json}"
.venv/bin/python scripts/check_dual_specialist_release_gate.py \
  --config "${ATR_DUAL_RELEASE_CONFIG}" \
  --output "${ATR_DUAL_RELEASE_OUTPUT}"

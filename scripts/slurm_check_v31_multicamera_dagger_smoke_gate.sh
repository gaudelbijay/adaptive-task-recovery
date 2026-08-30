#!/usr/bin/env bash
#SBATCH --job-name=atr-v31-camera-gate
#SBATCH --partition=compute-short
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=results/slurm/v31_camera_gate_%j.out
#SBATCH --error=results/slurm/v31_camera_gate_%j.err

set -euo pipefail
.venv/bin/python scripts/check_v31_multicamera_dagger_smoke_gate.py \
  --config configs/v31_multicamera_dagger_smoke_gate_v1.json \
  --output results/gates/v31_multicamera_dagger_smoke_gate_v1.json

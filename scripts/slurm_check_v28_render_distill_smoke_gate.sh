#!/usr/bin/env bash
#SBATCH --job-name=atr-v28-render-gate
#SBATCH --partition=compute-short
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=results/slurm/v28_render_gate_%j.out
#SBATCH --error=results/slurm/v28_render_gate_%j.err

set -euo pipefail
.venv/bin/python scripts/check_v27_robust_distill_smoke_gate.py \
  --config configs/v28_render_distill_smoke_gate_v1.json \
  --output results/gates/v28_render_distill_smoke_gate_v1.json

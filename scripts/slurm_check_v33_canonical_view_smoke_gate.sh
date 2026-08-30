#!/usr/bin/env bash
#SBATCH --job-name=atr-v33-canon-gate
#SBATCH --partition=compute-short
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=results/slurm/v33_canon_gate_%j.out
#SBATCH --error=results/slurm/v33_canon_gate_%j.err

set -euo pipefail
.venv/bin/python scripts/check_v33_canonical_view_smoke_gate.py \
  --config configs/v33_canonical_view_smoke_gate_v1.json \
  --output results/gates/v33_canonical_view_smoke_gate_v1.json

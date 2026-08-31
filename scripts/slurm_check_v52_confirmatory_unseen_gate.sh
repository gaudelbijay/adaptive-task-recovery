#!/usr/bin/env bash
#SBATCH --job-name=atr-v52-unseen-gate
#SBATCH --partition=compute-short
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=results/slurm/v52_unseen_gate_%j.out
#SBATCH --error=results/slurm/v52_unseen_gate_%j.err
set -euo pipefail
: "${ATR_V52_UNSEEN_GATE_CONFIG:?set ATR_V52_UNSEEN_GATE_CONFIG}"
mkdir -p results/slurm results/gates
.venv/bin/python scripts/check_v52_confirmatory_unseen_gate.py --config "$ATR_V52_UNSEEN_GATE_CONFIG" --output "${ATR_V52_UNSEEN_GATE_OUTPUT:-results/gates/v52_confirmatory_unseen_gate_v1.json}"

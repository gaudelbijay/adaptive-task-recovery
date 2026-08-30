#!/usr/bin/env bash
#SBATCH --job-name=atr-v3-competence-gate
#SBATCH --partition=compute-short
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --output=results/slurm/visual_competence_gate_%j.out
#SBATCH --error=results/slurm/visual_competence_gate_%j.err

set -euo pipefail
: "${ATR_GATE_AGGREGATE:?set ATR_GATE_AGGREGATE}"
: "${ATR_GATE_METHOD:?set ATR_GATE_METHOD}"
: "${ATR_RELEASE_JOB_ID:?set ATR_RELEASE_JOB_ID}"
ATR_PYTHON="${ATR_PYTHON:-.venv/bin/python}"

"${ATR_PYTHON}" scripts/check_visual_competence_gate.py \
  --aggregate "${ATR_GATE_AGGREGATE}" \
  --method "${ATR_GATE_METHOD}" \
  --minimum-success "${ATR_GATE_MINIMUM_SUCCESS:-0.70}" \
  --seeds "${ATR_GATE_SEEDS:-3}" \
  --episodes "${ATR_GATE_EPISODES:-768}"

scontrol release "${ATR_RELEASE_JOB_ID}"
echo "released ${ATR_RELEASE_JOB_ID} after held-out V3 competence gate"

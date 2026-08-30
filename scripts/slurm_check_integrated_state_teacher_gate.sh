#!/usr/bin/env bash
#SBATCH --job-name=atr-integrated-teacher-gate
#SBATCH --partition=compute-short
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --output=results/slurm/integrated_teacher_gate_%j.out
#SBATCH --error=results/slurm/integrated_teacher_gate_%j.err

set -euo pipefail
mkdir -p results/slurm results/gates
ATR_INTEGRATED_TEACHER_GATE_CONFIG="${ATR_INTEGRATED_TEACHER_GATE_CONFIG:-configs/integrated_state_teacher_gate_v1.json}"
ATR_INTEGRATED_TEACHER_GATE_OUTPUT="${ATR_INTEGRATED_TEACHER_GATE_OUTPUT:-results/gates/integrated_state_teacher_v1.json}"
.venv/bin/python scripts/check_integrated_state_teacher_gate.py \
  --config "${ATR_INTEGRATED_TEACHER_GATE_CONFIG}" \
  --output "${ATR_INTEGRATED_TEACHER_GATE_OUTPUT}"

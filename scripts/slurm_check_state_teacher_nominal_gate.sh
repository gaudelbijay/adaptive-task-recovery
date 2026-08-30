#!/usr/bin/env bash
#SBATCH --job-name=atr-state-teacher-gate
#SBATCH --partition=compute-short
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --output=results/slurm/state_teacher_gate_%j.out
#SBATCH --error=results/slurm/state_teacher_gate_%j.err

set -euo pipefail
mkdir -p results/slurm results/gates

.venv/bin/python scripts/check_state_teacher_nominal_gate.py \
  --aggregate results/learned_recovery/learned_recovery_ppo_v11_strict_removal/aggregate.json \
  --method event_reward_strict_removal_state_ppo \
  --output results/gates/strict_state_teacher_nominal_v1.json \
  --episodes 768 --seeds 3 \
  --minimum-raw 0.70 --minimum-safe 0.70 --maximum-violation 0.05

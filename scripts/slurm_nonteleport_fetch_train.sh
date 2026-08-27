#!/bin/bash
#SBATCH --job-name=atr-fetch-q
#SBATCH --partition=compute-v2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=23:50:00
#SBATCH --output=results/slurm/nonteleport_fetch_train_%j.out
#SBATCH --error=results/slurm/nonteleport_fetch_train_%j.err

set -euo pipefail
cd /home/bgaudel/adaptive-task-recovery
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
.venv/bin/python scripts/train_nonteleport_fetch_q.py \
  --episodes "${ATR_EPISODES:-30}" \
  --seed "${ATR_SEED:-0}" \
  --checkpoint-dir results/nonteleport_fetch/physical_q_cracker_v4_seed_${ATR_SEED:-0}/checkpoints

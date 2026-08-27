#!/bin/bash
#SBATCH --job-name=atr-fetch-eval
#SBATCH --partition=compute-v2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --array=0-89
#SBATCH --output=results/slurm/nonteleport_fetch_eval_%A_%a.out
#SBATCH --error=results/slurm/nonteleport_fetch_eval_%A_%a.err

set -euo pipefail
cd /home/bgaudel/adaptive-task-recovery
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

policies=(visual_learned_guarded static oracle)
policy="${policies[$((SLURM_ARRAY_TASK_ID % 3))]}"
seed="$((SLURM_ARRAY_TASK_ID / 3))"
.venv/bin/python scripts/run_nonteleport_pipeline.py \
  --checkpoint results/nonteleport_fetch/physical_q_cracker_v4_seed_0/checkpoints/latest.json \
  --recovery-change-threshold "${ATR_RECOVERY_CHANGE_THRESHOLD:?set calibrated threshold}" \
  --policy "${policy}" --seed "${seed}" --intervention cracker_box_destroyed \
  --output "results/nonteleport_fetch/eval/${policy}_seed_${seed}.json"

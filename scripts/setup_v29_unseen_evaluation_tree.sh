#!/usr/bin/env bash
#SBATCH --job-name=atr-v29-unseen-setup
#SBATCH --partition=compute-short
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --output=results/slurm/v29_unseen_setup_%j.out
#SBATCH --error=results/slurm/v29_unseen_setup_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source_root="$PWD/results/visual_recovery_ppo/visual_recovery_v19_multidomain_distill_v29/v19_multidomain_distillation"
output_root="$PWD/results/visual_recovery_ppo_v29_unseen"
test ! -e "${output_root}"
for seed in 9351 4796 1788; do
  source_dir="${source_root}/seed_${seed}"
  destination="${output_root}/visual_recovery_v19_multidomain_distill_v29/v19_multidomain_distillation/seed_${seed}"
  test -f "${source_dir}/best.pt"
  test -f "${source_dir}/TRAINING_COMPLETE.json"
  mkdir -p "${destination}"
  ln -s "${source_dir}/best.pt" "${destination}/best.pt"
  ln -s "${source_dir}/TRAINING_COMPLETE.json" "${destination}/TRAINING_COMPLETE.json"
done

#!/usr/bin/env bash
#SBATCH --job-name=atr-peg-specialist-eval
#SBATCH --partition=gpu-l40s
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --array=0-5
#SBATCH --output=results/slurm/peg_specialist_eval_%A_%a.out
#SBATCH --error=results/slurm/peg_specialist_eval_%A_%a.err

set -euo pipefail
read -r -a seeds <<< "${ATR_PEG_TRAINING_SEEDS:-9351 4796 1788}"
directions=(positive_ejection_recovery_specialist negative_ejection_recovery_specialist)
direction_index=$((SLURM_ARRAY_TASK_ID / 3))
seed_index=$((SLURM_ARRAY_TASK_ID % 3))
seed="${seeds[$seed_index]}"
direction="${directions[$direction_index]}"
root="${ATR_PEG_SPECIALIST_ROOT:-results/manipulation_ppo/external_peg_specialists_v1_directed_servo}"
output="${ATR_PEG_SPECIALIST_AUDIT_DIR:-results/a_plus_audit/external_peg_specialists_v1}"
mkdir -p results/slurm "${output}"
.venv/bin/python scripts/evaluate_external_peg_specialist.py \
  --checkpoint "${root}/${direction}/seed_${seed}/best.pt" \
  --episodes "${ATR_PEG_SPECIALIST_EVAL_EPISODES:-192}" \
  --num-envs "${ATR_PEG_SPECIALIST_EVAL_NUM_ENVS:-64}" \
  --steps 160 \
  --seed-base "$((421600000 + direction_index * 100000 + seed_index * 1000))" \
  --output "${output}/${direction}_seed_${seed}.json"

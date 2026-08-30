#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
smoke_job="${1:?usage: $0 SMOKE_ARRAY_JOB_ID}"
if [[ ! "${smoke_job}" =~ ^[0-9]+$ ]]; then
  echo "smoke job id must be numeric" >&2
  exit 2
fi
scontrol show job "${smoke_job}" >/dev/null

smoke_config=configs/visual_recovery_v19_render_distill_v28_smoke.json
full_config=configs/visual_recovery_v19_render_distill_v28.json
full_root=results/visual_recovery_ppo/visual_recovery_v19_render_distill_v28
if [[ -e "${full_root}" || -e results/visual_recovery_ppo_v28_unseen ]]; then
  echo "refusing duplicate V28 full/unseen submission" >&2
  exit 1
fi

smoke_audit_job=$(sbatch --parsable --dependency="afterok:${smoke_job}_*" \
  --export="ALL,ATR_AUDIT_CONFIG=${smoke_config},ATR_AUDIT_OUTPUT_ROOT=results/visual_recovery_ppo,ATR_AUDIT_FILE=results/visual_recovery_ppo/visual_recovery_v19_render_distill_v28_smoke/checkpoint_audit.json" \
  scripts/slurm_audit_training_checkpoints.sh)

development_job=$(sbatch --parsable --array=0-10%6 \
  --dependency="afterok:${smoke_audit_job}" \
  --export="ALL,ATR_V28_DEVELOPMENT_CONFIG=configs/v28_smoke_development_ood_v1.json" \
  scripts/slurm_v28_development_visual_ood.sh)

development_aggregate_job=$(sbatch --parsable --partition=compute-short \
  --cpus-per-task=2 --mem=16G --time=00:30:00 \
  --job-name=atr-v28-dev-aggregate \
  --output=results/slurm/v28_dev_aggregate_%j.out \
  --error=results/slurm/v28_dev_aggregate_%j.err \
  --dependency="afterok:${development_job}_*" \
  --wrap=".venv/bin/python scripts/aggregate_selected_visual_causal_ood.py --config configs/v28_smoke_development_ood_v1.json --results-root results/visual_recovery_ppo --output results/paper/v28_smoke_development_ood_v1/aggregate.json")

allocation_gate_job=$(sbatch --parsable \
  --dependency="afterok:${development_aggregate_job}" \
  scripts/slurm_check_v28_render_distill_smoke_gate.sh)

full_job=$(sbatch --parsable --array=0-2%3 \
  --dependency="afterok:${allocation_gate_job}" \
  --export="ALL,ATR_RENDER_DISTILL_CONFIG=${full_config}" \
  scripts/slurm_v19_rendered_domain_distillation.sh)

audit_job=$(sbatch --parsable --dependency="afterok:${full_job}_*" \
  --export="ALL,ATR_AUDIT_CONFIG=${full_config},ATR_AUDIT_OUTPUT_ROOT=results/visual_recovery_ppo,ATR_AUDIT_FILE=${full_root}/checkpoint_audit.json" \
  scripts/slurm_audit_training_checkpoints.sh)

standard_job=$(sbatch --parsable --array=0-2%3 \
  --dependency="afterok:${audit_job}" \
  --export="ALL,ATR_VISUAL_CONFIG=${full_config}" \
  scripts/slurm_v28_visual_recovery_eval.sh)

standard_aggregate_job=$(sbatch --parsable \
  --dependency="afterok:${standard_job}_*" \
  --export="ALL,ATR_VISUAL_CONFIG=${full_config},ATR_AGGREGATE_FILENAME=aggregate.json" \
  scripts/slurm_visual_recovery_aggregate.sh)

strict_job=$(sbatch --parsable --array=0-2%3 \
  --dependency="afterok:${audit_job}" \
  --export="ALL,ATR_VISUAL_CONFIG=${full_config}" \
  scripts/slurm_v28_visual_strict_removal_eval.sh)

strict_aggregate_job=$(sbatch --parsable \
  --dependency="afterok:${strict_job}_*" \
  --export="ALL,ATR_STRICT_COMPARISON_CONFIG=configs/strict_removal_v19_render_distill_v16.json" \
  scripts/slurm_aggregate_strict_removal.sh)

unseen_setup_job=$(sbatch --parsable --dependency="afterok:${audit_job}" \
  scripts/setup_v28_unseen_evaluation_tree.sh)

unseen_job=$(sbatch --parsable --array=0-26%8 \
  --dependency="afterok:${unseen_setup_job}" \
  --export="ALL,ATR_V28_UNSEEN_CONFIG=configs/v28_unseen_visual_ood_v1.json,ATR_VISUAL_OUTPUT=results/visual_recovery_ppo_v28_unseen" \
  scripts/slurm_v28_unseen_visual_ood.sh)

unseen_aggregate_job=$(sbatch --parsable --partition=compute-short \
  --cpus-per-task=2 --mem=16G --time=00:30:00 \
  --job-name=atr-v28-unseen-aggregate \
  --output=results/slurm/v28_unseen_aggregate_%j.out \
  --error=results/slurm/v28_unseen_aggregate_%j.err \
  --dependency="afterok:${unseen_job}_*" \
  --wrap=".venv/bin/python scripts/aggregate_selected_visual_causal_ood.py --config configs/v28_unseen_visual_ood_v1.json --results-root results/visual_recovery_ppo_v28_unseen --output results/paper/v28_unseen_visual_ood_v1/aggregate.json")

final_gate_job=$(sbatch --parsable \
  --dependency="afterok:${standard_aggregate_job},afterok:${strict_aggregate_job},afterok:${unseen_aggregate_job}" \
  scripts/slurm_check_v28_final_release_gate.sh)

printf '%s\n' \
  "V28 smoke=${smoke_job}" \
  "V28 smoke audit=${smoke_audit_job}" \
  "V28 development OOD=${development_job}" \
  "V28 development aggregate=${development_aggregate_job}" \
  "V28 allocation gate=${allocation_gate_job}" \
  "V28 full=${full_job}" \
  "V28 audit=${audit_job}" \
  "V28 standard=${standard_job}" \
  "V28 standard aggregate=${standard_aggregate_job}" \
  "V28 strict=${strict_job}" \
  "V28 strict aggregate=${strict_aggregate_job}" \
  "V28 unseen setup=${unseen_setup_job}" \
  "V28 unseen=${unseen_job}" \
  "V28 unseen aggregate=${unseen_aggregate_job}" \
  "V28 final gate=${final_gate_job}"

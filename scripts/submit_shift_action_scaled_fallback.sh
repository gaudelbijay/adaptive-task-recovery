#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

V24_GATE_RESULT="results/gates/shift_action_stability_smoke_gate_v1.json"
V24_GATE_CONFIG="configs/shift_action_stability_smoke_gate_v1.json"
V25_ROUTE_RESULT="results/gates/shift_action_scaled_fallback_route_v1.json"
V25_SMOKE_CONFIG="configs/visual_recovery_dual_specialist_shift_action_v25_scaled_smoke.json"
V25_FULL_CONFIG="configs/visual_recovery_dual_specialist_shift_action_v25_scaled.json"
V25_RUN_ROOT="results/visual_recovery_ppo/visual_recovery_dual_specialist_shift_action_v25_scaled"
V25_SMOKE_ROOT="results/visual_recovery_ppo/visual_recovery_dual_specialist_shift_action_v25_scaled_smoke"

# This exits nonzero for a V24 pass or for any missing/malformed/inconsistent
# evidence. No sbatch call occurs unless V24 was explicitly and validly rejected.
.venv/bin/python scripts/route_shift_action_scaled_fallback.py \
  --upstream-result "${V24_GATE_RESULT}" \
  --gate-config "${V24_GATE_CONFIG}" \
  --output "${V25_ROUTE_RESULT}"

if [[ -e "${V25_RUN_ROOT}" || -e "${V25_SMOKE_ROOT}" ]]; then
  echo "refusing duplicate V25 submission: result directory already exists" >&2
  exit 1
fi

smoke_job=$(sbatch --parsable --array=0 \
  --export="ALL,ATR_VISUAL_CONFIG=${V25_SMOKE_CONFIG}" \
  scripts/slurm_visual_recovery_dual_teacher_shift_action_ppo.sh)

gate_job=$(sbatch --parsable --dependency="afterok:${smoke_job}" \
  scripts/slurm_check_shift_action_scaled_stability_smoke_gate.sh)

full_job=$(sbatch --parsable --array=0-2%3 \
  --dependency="afterok:${gate_job}" \
  --export="ALL,ATR_VISUAL_CONFIG=${V25_FULL_CONFIG}" \
  scripts/slurm_visual_recovery_dual_teacher_shift_action_ppo.sh)

audit_job=$(sbatch --parsable --dependency="afterok:${full_job}_*" \
  --export="ALL,ATR_AUDIT_CONFIG=${V25_FULL_CONFIG},ATR_AUDIT_OUTPUT_ROOT=results/visual_recovery_ppo,ATR_AUDIT_FILE=${V25_RUN_ROOT}/checkpoint_audit.json" \
  scripts/slurm_audit_training_checkpoints.sh)

strict_job=$(sbatch --parsable --array=0-2%3 \
  --dependency="afterok:${audit_job}" \
  --export="ALL,ATR_VISUAL_CONFIG=${V25_FULL_CONFIG}" \
  scripts/slurm_visual_strict_removal_eval.sh)

nominal_job=$(sbatch --parsable --array=0-2%3 \
  --dependency="afterok:${audit_job}" \
  --export="ALL,ATR_VISUAL_CONFIG=${V25_FULL_CONFIG},ATR_EVAL_CONDITIONS=nominal" \
  scripts/slurm_visual_recovery_eval.sh)

nominal_aggregate_job=$(sbatch --parsable \
  --dependency="afterok:${nominal_job}_*" \
  --export="ALL,ATR_VISUAL_CONFIG=${V25_FULL_CONFIG},ATR_AGGREGATE_CONDITIONS=nominal,ATR_AGGREGATE_FILENAME=aggregate_nominal.json" \
  scripts/slurm_visual_recovery_aggregate.sh)

strict_aggregate_job=$(sbatch --parsable \
  --dependency="afterok:${strict_job}_*,afterok:1140386" \
  --export="ALL,ATR_STRICT_COMPARISON_CONFIG=configs/strict_removal_dual_specialist_shift_action_scaled_extension_v13.json" \
  scripts/slurm_aggregate_strict_removal.sh)

selector_job=$(sbatch --parsable \
  --dependency="afterok:${strict_aggregate_job},afterok:${nominal_aggregate_job},afterok:1140387" \
  --export="ALL,ATR_SELECTION_CONFIG=configs/integrated_visual_selection_v10.json,ATR_SELECTION_OUTPUT=results/gates/integrated_visual_selection_v10.json" \
  scripts/slurm_select_integrated_visual_policy.sh)

causal_job=$(sbatch --parsable --array=0-29%6 \
  --dependency="afterok:${selector_job}" \
  --export="ALL,ATR_ABLATION_CONFIG=configs/selected_visual_causal_ood_v2.json" \
  scripts/slurm_selected_visual_causal_ood.sh)

causal_aggregate_job=$(sbatch --parsable --partition=compute-short \
  --cpus-per-task=2 --mem=16G --time=00:30:00 \
  --job-name=atr-causal-ood-aggregate-v2 \
  --output=results/slurm/causal_ood_aggregate_%j.out \
  --error=results/slurm/causal_ood_aggregate_%j.err \
  --dependency="afterok:${causal_job}_*" \
  --wrap=".venv/bin/python scripts/aggregate_selected_visual_causal_ood.py --config configs/selected_visual_causal_ood_v2.json --results-root results/visual_recovery_ppo --output results/paper/selected_visual_causal_ood_v2/aggregate.json")

printf '%s\n' \
  "V25 smoke=${smoke_job}" \
  "V25 gate=${gate_job}" \
  "V25 full=${full_job}" \
  "V25 audit=${audit_job}" \
  "V25 strict=${strict_job}" \
  "V25 nominal=${nominal_job}" \
  "V25 nominal aggregate=${nominal_aggregate_job}" \
  "V25 strict aggregate=${strict_aggregate_job}" \
  "V25 selector=${selector_job}" \
  "V25 causal/OOD=${causal_job}" \
  "V25 causal/OOD aggregate=${causal_aggregate_job}"

#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
gate_job="${1:?usage: $0 COMPLETED_V29_GATE_JOB_ID}"
if [[ ! "${gate_job}" =~ ^[0-9]+$ ]]; then
  echo "gate job id must be numeric" >&2
  exit 2
fi
gate_result=results/gates/v29_multidomain_distill_smoke_gate_v1.json
.venv/bin/python - "${gate_result}" <<'PY'
import hashlib, json, sys
from pathlib import Path
result = json.loads(Path(sys.argv[1]).read_text())
config = Path("configs/v29_multidomain_distill_smoke_gate_v1.json")
if result.get("eligible") is not True:
    raise SystemExit("V29 allocation gate did not pass")
if result.get("source_sha256", {}).get(str(config)) != hashlib.sha256(config.read_bytes()).hexdigest():
    raise SystemExit("V29 allocation gate/config provenance mismatch")
PY

full_config=configs/visual_recovery_v19_multidomain_distill_v29.json
full_root=results/visual_recovery_ppo/visual_recovery_v19_multidomain_distill_v29
if [[ -e "${full_root}" || -e results/visual_recovery_ppo_v29_unseen ]]; then
  echo "refusing duplicate V29 full/unseen submission" >&2
  exit 1
fi

full_job=$(sbatch --parsable --array=0-2%3 --dependency="afterok:${gate_job}" \
  --export="ALL,ATR_MULTIDOMAIN_DISTILL_CONFIG=${full_config}" \
  scripts/slurm_v19_multidomain_distillation.sh)
audit_job=$(sbatch --parsable --dependency="afterok:${full_job}_*" \
  --export="ALL,ATR_AUDIT_CONFIG=${full_config},ATR_AUDIT_OUTPUT_ROOT=results/visual_recovery_ppo,ATR_AUDIT_FILE=${full_root}/checkpoint_audit.json" \
  scripts/slurm_audit_training_checkpoints.sh)
standard_job=$(sbatch --parsable --array=0-2%3 --dependency="afterok:${audit_job}" \
  --export="ALL,ATR_VISUAL_CONFIG=${full_config}" scripts/slurm_v29_visual_recovery_eval.sh)
standard_aggregate_job=$(sbatch --parsable --dependency="afterok:${standard_job}_*" \
  --export="ALL,ATR_VISUAL_CONFIG=${full_config},ATR_AGGREGATE_FILENAME=aggregate.json" \
  scripts/slurm_visual_recovery_aggregate.sh)
strict_job=$(sbatch --parsable --array=0-2%3 --dependency="afterok:${audit_job}" \
  --export="ALL,ATR_VISUAL_CONFIG=${full_config}" scripts/slurm_v29_visual_strict_removal_eval.sh)
strict_aggregate_job=$(sbatch --parsable --dependency="afterok:${strict_job}_*" \
  --export="ALL,ATR_STRICT_COMPARISON_CONFIG=configs/strict_removal_v19_multidomain_distill_v17.json" \
  scripts/slurm_aggregate_strict_removal.sh)
unseen_setup_job=$(sbatch --parsable --dependency="afterok:${audit_job}" \
  scripts/setup_v29_unseen_evaluation_tree.sh)
unseen_job=$(sbatch --parsable --array=0-26%8 --dependency="afterok:${unseen_setup_job}" \
  --export="ALL,ATR_V29_UNSEEN_CONFIG=configs/v29_unseen_visual_ood_v1.json,ATR_VISUAL_OUTPUT=results/visual_recovery_ppo_v29_unseen" \
  scripts/slurm_v29_unseen_visual_ood.sh)
unseen_aggregate_job=$(sbatch --parsable --partition=compute-short \
  --cpus-per-task=2 --mem=16G --time=00:30:00 \
  --job-name=atr-v29-unseen-aggregate \
  --output=results/slurm/v29_unseen_aggregate_%j.out \
  --error=results/slurm/v29_unseen_aggregate_%j.err \
  --dependency="afterok:${unseen_job}_*" \
  --wrap=".venv/bin/python scripts/aggregate_selected_visual_causal_ood.py --config configs/v29_unseen_visual_ood_v1.json --results-root results/visual_recovery_ppo_v29_unseen --output results/paper/v29_unseen_visual_ood_v1/aggregate.json")
final_gate_job=$(sbatch --parsable \
  --dependency="afterok:${standard_aggregate_job},afterok:${strict_aggregate_job},afterok:${unseen_aggregate_job}" \
  scripts/slurm_check_v29_final_release_gate.sh)

printf '%s\n' \
  "V29 full=${full_job}" "V29 audit=${audit_job}" \
  "V29 standard=${standard_job}" "V29 standard aggregate=${standard_aggregate_job}" \
  "V29 strict=${strict_job}" "V29 strict aggregate=${strict_aggregate_job}" \
  "V29 unseen setup=${unseen_setup_job}" "V29 unseen=${unseen_job}" \
  "V29 unseen aggregate=${unseen_aggregate_job}" "V29 final gate=${final_gate_job}"

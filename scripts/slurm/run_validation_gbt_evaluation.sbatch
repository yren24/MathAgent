#!/bin/bash
#SBATCH --job-name=mint-agent-validation-gbt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

set -euo pipefail

MINT_AGENT_ROOT=${MINT_AGENT_ROOT:?MINT_AGENT_ROOT must be provided}
TASK_CONFIG=${TASK_CONFIG:?TASK_CONFIG must be provided}
REPRESENTATION_SPEC=${REPRESENTATION_SPEC:?REPRESENTATION_SPEC must be provided}
GBT_CONFIG=${GBT_CONFIG:?GBT_CONFIG must be provided}
SCOUT_ARTIFACT=${SCOUT_ARTIFACT:?SCOUT_ARTIFACT must be provided}
TRAIN_QC_REPORT=${TRAIN_QC_REPORT:?TRAIN_QC_REPORT must be provided}
VALIDATION_QC_REPORT=${VALIDATION_QC_REPORT:?VALIDATION_QC_REPORT must be provided}
PRIOR_EVALUATION_REPORT=${PRIOR_EVALUATION_REPORT:-}
FEATURE_INVARIANTS=${FEATURE_INVARIANTS:?FEATURE_INVARIANTS must be provided}
MAX_ACQUISITIONS=${MAX_ACQUISITIONS:?MAX_ACQUISITIONS must be provided}
OUTPUT=${OUTPUT:?OUTPUT must be provided}
PYTHON_BIN=${PYTHON_BIN:-python}

cd "${MINT_AGENT_ROOT}"
export PYTHONPATH=${MINT_AGENT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}

if [[ -n "${MINT_MODULE_SETUP:-}" ]]; then
  source "${MINT_MODULE_SETUP}"
elif [[ -n "${MINT_MODULES:-}" ]]; then
  module purge
  IFS=':' read -r -a mint_modules <<< "${MINT_MODULES}"
  for module_name in "${mint_modules[@]}"; do
    module load "${module_name}"
  done
fi

args=(
  --task-config "${TASK_CONFIG}"
  --representation-spec "${REPRESENTATION_SPEC}"
  --gbt-config "${GBT_CONFIG}"
  --scout-artifact "${SCOUT_ARTIFACT}"
  --train-feature-qc-report "${TRAIN_QC_REPORT}"
  --validation-feature-qc-report "${VALIDATION_QC_REPORT}"
  --max-acquisitions "${MAX_ACQUISITIONS}"
  --output "${OUTPUT}"
)
[[ -n "${PRIOR_EVALUATION_REPORT}" ]] && args+=(--prior-evaluation-report "${PRIOR_EVALUATION_REPORT}")

IFS=':' read -r -a feature_invariants <<< "${FEATURE_INVARIANTS}"
for invariant in "${feature_invariants[@]}"; do
  for split in TRAIN VALIDATION; do
    variable_name="FEATURE_MANIFEST_${split}_${invariant}"
    manifest=${!variable_name:-}
    if [[ -z "${manifest}" ]]; then
      echo "Missing ${variable_name}" >&2
      exit 2
    fi
    option_name=$(printf '%s' "${split}" | tr '[:upper:]' '[:lower:]')
    args+=("--${option_name}-feature-manifest" "${invariant}=${manifest}")
  done
done

"${PYTHON_BIN}" -m mint_scout.evaluate_validation_gbt "${args[@]}"

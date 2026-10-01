#!/bin/bash
#SBATCH --job-name=mint-agent-scout-oof
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

set -euo pipefail

MINT_AGENT_ROOT=${MINT_AGENT_ROOT:?MINT_AGENT_ROOT must be provided by the execution profile}
RUN_ROOT=${RUN_ROOT:-${MINT_AGENT_ROOT}/runs}
LOG_ROOT=${LOG_ROOT:-${MINT_AGENT_ROOT}/logs}
PYTHON_BIN=${PYTHON_BIN:-python}
DATASET_ID=${DATASET_ID:?DATASET_ID must be provided}
TASK_CONFIG=${TASK_CONFIG:?TASK_CONFIG must be provided}
SCOUT_CONFIG=${SCOUT_CONFIG:?SCOUT_CONFIG must be provided}
GBT_CONFIG=${GBT_CONFIG:-}
REPRESENTATION_SPEC=${REPRESENTATION_SPEC:?REPRESENTATION_SPEC must be provided}
PROBE_SELECTION=${PROBE_SELECTION:?PROBE_SELECTION must be provided}
FEATURE_QC_REPORT=${FEATURE_QC_REPORT:?FEATURE_QC_REPORT must be provided}
FILTRATION_AUDIT_REPORT=${FILTRATION_AUDIT_REPORT:?FILTRATION_AUDIT_REPORT must be provided}
FEATURE_INVARIANTS=${FEATURE_INVARIANTS:?FEATURE_INVARIANTS must be provided}
OUTPUT=${OUTPUT:?OUTPUT must be provided}
EXECUTION_ARTIFACT=${EXECUTION_ARTIFACT:?EXECUTION_ARTIFACT must be provided}

mkdir -p "${LOG_ROOT}" "${RUN_ROOT}"
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
  --scout-config "${SCOUT_CONFIG}"
  --representation-spec "${REPRESENTATION_SPEC}"
  --probe-selection "${PROBE_SELECTION}"
  --feature-qc-report "${FEATURE_QC_REPORT}"
  --filtration-audit-report "${FILTRATION_AUDIT_REPORT}"
  --output "${OUTPUT}"
  --execution-artifact "${EXECUTION_ARTIFACT}"
)
if [[ -n "${GBT_CONFIG}" ]]; then
  args+=(--gbt-config "${GBT_CONFIG}")
fi
IFS=':' read -r -a feature_invariants <<< "${FEATURE_INVARIANTS}"
for invariant in "${feature_invariants[@]}"; do
  variable_name="FEATURE_MANIFEST_${invariant}"
  manifest=${!variable_name:-}
  if [[ -z "${manifest}" ]]; then
    echo "Missing ${variable_name}" >&2
    exit 2
  fi
  args+=(--feature-manifest "${invariant}=${manifest}")
done
if [[ -n "${USER_TARGET:-}" ]]; then
  args+=(--user-target "${USER_TARGET}")
fi

"${PYTHON_BIN}" -m mint_scout.run_casf_scout "${args[@]}"

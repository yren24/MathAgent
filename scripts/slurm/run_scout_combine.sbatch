#!/bin/bash
#SBATCH --job-name=mint-agent-scout-combine
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

set -euo pipefail

MINT_AGENT_ROOT=${MINT_AGENT_ROOT:?MINT_AGENT_ROOT must be provided by the execution profile}
RUN_ROOT=${RUN_ROOT:-${MINT_AGENT_ROOT}/runs}
LOG_ROOT=${LOG_ROOT:-${MINT_AGENT_ROOT}/logs}
PYTHON_BIN=${PYTHON_BIN:-python}
DATASET_ID=${DATASET_ID:?DATASET_ID must be provided}
SCOUT_CONFIG=${SCOUT_CONFIG:?SCOUT_CONFIG must be provided}
REPRESENTATION_SPEC=${REPRESENTATION_SPEC:?REPRESENTATION_SPEC must be provided}
PROBE_SELECTION=${PROBE_SELECTION:?PROBE_SELECTION must be provided}
FEATURE_QC_REPORT=${FEATURE_QC_REPORT:?FEATURE_QC_REPORT must be provided}
RANKING_POLICY=${RANKING_POLICY:?RANKING_POLICY must be provided}
SOURCE_INVARIANTS=${SOURCE_INVARIANTS:?SOURCE_INVARIANTS must be provided}
COMBINED_REPORT=${COMBINED_REPORT:?COMBINED_REPORT must be provided}
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
  --dataset-id "${DATASET_ID}"
  --scout-config "${SCOUT_CONFIG}"
  --representation-spec "${REPRESENTATION_SPEC}"
  --probe-selection "${PROBE_SELECTION}"
  --feature-qc-report "${FEATURE_QC_REPORT}"
  --ranking-policy "${RANKING_POLICY}"
  --output "${COMBINED_REPORT}"
  --execution-artifact "${EXECUTION_ARTIFACT}"
)
IFS=':' read -r -a source_invariants <<< "${SOURCE_INVARIANTS}"
for invariant in "${source_invariants[@]}"; do
  variable_name="SCOUT_REPORT_${invariant}"
  report=${!variable_name:-}
  if [[ -z "${report}" ]]; then
    echo "Missing ${variable_name}" >&2
    exit 2
  fi
  args+=(--scout-report "${report}")
done
if [[ -n "${USER_TARGET:-}" ]]; then
  args+=(--user-target "${USER_TARGET}")
fi
if [[ -n "${LLM_PRIOR_CANDIDATES:-}" ]]; then
  IFS=';' read -r -a llm_prior_candidates <<< "${LLM_PRIOR_CANDIDATES}"
  for candidate in "${llm_prior_candidates[@]}"; do
    args+=(--llm-prior-candidate "${candidate}")
  done
fi

"${PYTHON_BIN}" -m mint_scout.combine_scout_oof "${args[@]}"

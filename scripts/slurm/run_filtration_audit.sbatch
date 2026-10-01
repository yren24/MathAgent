#!/bin/bash
#SBATCH --job-name=mint-agent-filtration-audit
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

set -euo pipefail

MINT_AGENT_ROOT=${MINT_AGENT_ROOT:?MINT_AGENT_ROOT must be provided by the execution profile}
RUN_ROOT=${RUN_ROOT:-${MINT_AGENT_ROOT}/runs}
LOG_ROOT=${LOG_ROOT:-${MINT_AGENT_ROOT}/logs}
PYTHON_BIN=${PYTHON_BIN:-python}
DATASET_ID=${DATASET_ID:?DATASET_ID must be provided}
EVIDENCE_SCOPE=${EVIDENCE_SCOPE:?EVIDENCE_SCOPE must be provided}
REPRESENTATION_SPEC=${REPRESENTATION_SPEC:?REPRESENTATION_SPEC must be provided}
SAMPLE_ID_FILE=${SAMPLE_ID_FILE:-}
PROBE_SELECTION=${PROBE_SELECTION:-}
AUDIT_CONFIG=${AUDIT_CONFIG:?AUDIT_CONFIG must be provided}
FEATURE_INVARIANTS=${FEATURE_INVARIANTS:?FEATURE_INVARIANTS must be provided}
OUTPUT=${OUTPUT:-${RUN_ROOT}/filtration_audit_${SLURM_JOB_ID:-manual}.json}

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
  --representation-spec "${REPRESENTATION_SPEC}"
  --audit-config "${AUDIT_CONFIG}"
  --dataset-id "${DATASET_ID}"
  --evidence-scope "${EVIDENCE_SCOPE}"
  --output "${OUTPUT}"
)
if [[ -n "${PROBE_SELECTION}" ]]; then
  args+=(--probe-selection "${PROBE_SELECTION}")
elif [[ -n "${SAMPLE_ID_FILE}" ]]; then
  args+=(--sample-id-file "${SAMPLE_ID_FILE}")
else
  echo "PROBE_SELECTION or SAMPLE_ID_FILE must be provided" >&2
  exit 2
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

# A scientifically failed audit is still a successfully produced audit artifact.
set +e
"${PYTHON_BIN}" -m mint_scout.filtration_audit "${args[@]}"
audit_exit=$?
set -e
if [[ -s "${OUTPUT}" ]]; then
  exit 0
fi
exit "${audit_exit}"

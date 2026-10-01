#!/bin/bash
#SBATCH --job-name=mint-agent-feature-diagnostic
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
FEATURE_INVARIANTS=${FEATURE_INVARIANTS:?FEATURE_INVARIANTS must be provided}
FEATURE_QC_REPORT=${FEATURE_QC_REPORT:?FEATURE_QC_REPORT must be provided}
TOP_K=${TOP_K:-10}
OUTPUT=${OUTPUT:-${RUN_ROOT}/feature_outlier_diagnostic_${SLURM_JOB_ID:-manual}.json}

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

IFS=':' read -r -a feature_invariants <<< "${FEATURE_INVARIANTS}"
if [[ ${#feature_invariants[@]} -ne 1 ]]; then
  echo "Feature outlier diagnosis requires exactly one invariant" >&2
  exit 2
fi
invariant=${feature_invariants[0]}
variable_name="FEATURE_MANIFEST_${invariant}"
feature_manifest=${!variable_name:-}
if [[ -z "${feature_manifest}" ]]; then
  echo "Missing ${variable_name}" >&2
  exit 2
fi

args=(
  --representation-spec "${REPRESENTATION_SPEC}"
  --feature-manifest "${invariant}=${feature_manifest}"
  --qc-report "${FEATURE_QC_REPORT}"
  --top-k "${TOP_K}"
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

"${PYTHON_BIN}" -m mint_scout.feature_outlier_diagnostic "${args[@]}"

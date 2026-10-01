#!/bin/bash
#SBATCH --job-name=mint-agent-feature-qc
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
TASK_CONFIG=${TASK_CONFIG:-}
SPLIT=${SPLIT:-}
QC_CONFIG=${QC_CONFIG:?QC_CONFIG must be provided}
FEATURE_INVARIANTS=${FEATURE_INVARIANTS:?FEATURE_INVARIANTS must be provided}
OUTPUT=${OUTPUT:-${RUN_ROOT}/feature_qc_${SLURM_JOB_ID:-manual}.json}

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

if [[ "${MINT_DEBUG_ENV:-0}" == "1" ]]; then
  echo "MINT_MODULE_SETUP=${MINT_MODULE_SETUP:-<unset>}"
  echo "MINT_MODULES=${MINT_MODULES:-<unset>}"
  echo "PYTHON_BIN=$(command -v "${PYTHON_BIN}" || true)"
  echo "EBPYTHONPREFIXES=${EBPYTHONPREFIXES:-<unset>}"
  echo "PYTHONPATH=${PYTHONPATH}"
  module -t list 2>&1 || true
fi

args=(
  --representation-spec "${REPRESENTATION_SPEC}"
  --qc-config "${QC_CONFIG}"
  --dataset-id "${DATASET_ID}"
  --evidence-scope "${EVIDENCE_SCOPE}"
  --output "${OUTPUT}"
)
if [[ -n "${PROBE_SELECTION}" ]]; then
  args+=(--probe-selection "${PROBE_SELECTION}")
elif [[ -n "${SAMPLE_ID_FILE}" ]]; then
  args+=(--sample-id-file "${SAMPLE_ID_FILE}")
elif [[ -n "${TASK_CONFIG}" && -n "${SPLIT}" ]]; then
  args+=(--task-config "${TASK_CONFIG}" --split "${SPLIT}")
else
  echo "PROBE_SELECTION, SAMPLE_ID_FILE, or TASK_CONFIG plus SPLIT must be provided" >&2
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

# A scientifically failed QC is still a successfully produced audit artifact.
set +e
"${PYTHON_BIN}" -m mint_scout.feature_qc "${args[@]}"
qc_exit=$?
set -e
if [[ -s "${OUTPUT}" ]]; then
  exit 0
fi
exit "${qc_exit}"

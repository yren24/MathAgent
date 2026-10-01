#!/bin/bash
#SBATCH --job-name=mint-agent-model-evaluation
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

set -euo pipefail

MINT_AGENT_ROOT=${MINT_AGENT_ROOT:?MINT_AGENT_ROOT must be provided by the execution profile}
RUN_ROOT=${RUN_ROOT:-${MINT_AGENT_ROOT}/runs}
LOG_ROOT=${LOG_ROOT:-${MINT_AGENT_ROOT}/logs}
PYTHON_BIN=${PYTHON_BIN:-python}
DATASET_ID=${DATASET_ID:?DATASET_ID must be provided}
EVIDENCE_SCOPE=${EVIDENCE_SCOPE:?EVIDENCE_SCOPE must be provided}
TASK_CONFIG=${TASK_CONFIG:?TASK_CONFIG must be provided}
GBT_CONFIG=${GBT_CONFIG:?GBT_CONFIG must be provided}
REPRESENTATION_SPEC=${REPRESENTATION_SPEC:?REPRESENTATION_SPEC must be provided}
SCOUT_ARTIFACT=${SCOUT_ARTIFACT:?SCOUT_ARTIFACT must be provided}
FEATURE_QC_REPORT=${FEATURE_QC_REPORT:?FEATURE_QC_REPORT must be provided}
FEATURE_INVARIANTS=${FEATURE_INVARIANTS:?FEATURE_INVARIANTS must be provided}
MAX_ACQUISITIONS=${MAX_ACQUISITIONS:?MAX_ACQUISITIONS must be provided}
OUTPUT=${OUTPUT:?OUTPUT must be provided}

if [[ "${EVIDENCE_SCOPE}" != "full_train" ]]; then
  echo "V1 model-evaluation runner supports full_train only" >&2
  exit 2
fi

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
  --config "${TASK_CONFIG}"
  --gbt-config "${GBT_CONFIG}"
  --split train
  --scout-artifact "${SCOUT_ARTIFACT}"
  --representation-spec "${REPRESENTATION_SPEC}"
  --feature-qc-report "${FEATURE_QC_REPORT}"
  --max-acquisitions "${MAX_ACQUISITIONS}"
  --output "${OUTPUT}"
)
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

"${PYTHON_BIN}" -m mint_scout.execute_casf "${args[@]}"

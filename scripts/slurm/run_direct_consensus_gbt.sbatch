#!/bin/bash
#SBATCH --job-name=mint-direct-consensus-gbt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

set -euo pipefail

MINT_AGENT_ROOT=${MINT_AGENT_ROOT:?MINT_AGENT_ROOT must be provided}
TASK_CONFIG=${TASK_CONFIG:?TASK_CONFIG must be provided}
GBT_CONFIG=${GBT_CONFIG:?GBT_CONFIG must be provided}
REPRESENTATION_SPEC=${REPRESENTATION_SPEC:?REPRESENTATION_SPEC must be provided}
FEATURE_INVARIANTS=${FEATURE_INVARIANTS:?FEATURE_INVARIANTS must be provided}
OUTPUT=${OUTPUT:?OUTPUT must be provided}
LOG_ROOT=${LOG_ROOT:-${MINT_AGENT_ROOT}/logs}
PYTHON_BIN=${PYTHON_BIN:-python}

mkdir -p "${LOG_ROOT}" "$(dirname "${OUTPUT}")"
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
  --output "${OUTPUT}"
)
IFS=':' read -r -a invariants <<< "${FEATURE_INVARIANTS}"
for invariant in "${invariants[@]}"; do
  train_var="FEATURE_MANIFEST_TRAIN_${invariant}"
  test_var="FEATURE_MANIFEST_TEST_${invariant}"
  train_manifest=${!train_var:-}
  test_manifest=${!test_var:-}
  if [[ -z "${train_manifest}" || -z "${test_manifest}" ]]; then
    echo "Missing ${train_var} or ${test_var}" >&2
    exit 2
  fi
  args+=(--train-feature-manifest "${invariant}=${train_manifest}")
  args+=(--test-feature-manifest "${invariant}=${test_manifest}")
done
if [[ -n "${PRIOR_REPORTS:-}" ]]; then
  IFS=':' read -r -a prior_reports <<< "${PRIOR_REPORTS}"
  for prior_report in "${prior_reports[@]}"; do
    args+=(--prior-report "${prior_report}")
  done
fi

"${PYTHON_BIN}" -m mint_scout.evaluate_direct_consensus_gbt "${args[@]}"

#!/bin/bash
#SBATCH --job-name=mint-agent-split-gbt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

set -euo pipefail

MINT_AGENT_ROOT=${MINT_AGENT_ROOT:?MINT_AGENT_ROOT must be provided}
TASK_CONFIG=${TASK_CONFIG:?TASK_CONFIG must be provided}
REPRESENTATION_SPEC=${REPRESENTATION_SPEC:?REPRESENTATION_SPEC must be provided}
GBT_CONFIG=${GBT_CONFIG:?GBT_CONFIG must be provided}
TRAIN_QC_REPORT=${TRAIN_QC_REPORT:?TRAIN_QC_REPORT must be provided}
VALIDATION_QC_REPORT=${VALIDATION_QC_REPORT:?VALIDATION_QC_REPORT must be provided}
TEST_QC_REPORT=${TEST_QC_REPORT:?TEST_QC_REPORT must be provided}
FEATURE_INVARIANTS=${FEATURE_INVARIANTS:?FEATURE_INVARIANTS must be provided}
OUTPUT=${OUTPUT:?OUTPUT must be provided}
PYTHON_BIN=${PYTHON_BIN:-python}
N_BOOTSTRAP=${N_BOOTSTRAP:-1000}
BOOTSTRAP_SEED=${BOOTSTRAP_SEED:-2026}

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
  --train-feature-qc-report "${TRAIN_QC_REPORT}"
  --validation-feature-qc-report "${VALIDATION_QC_REPORT}"
  --test-feature-qc-report "${TEST_QC_REPORT}"
  --n-bootstrap "${N_BOOTSTRAP}"
  --bootstrap-seed "${BOOTSTRAP_SEED}"
  --output "${OUTPUT}"
)

IFS=':' read -r -a feature_invariants <<< "${FEATURE_INVARIANTS}"
for invariant in "${feature_invariants[@]}"; do
  for split in TRAIN VALIDATION TEST; do
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

"${PYTHON_BIN}" -m mint_scout.evaluate_split_gbt "${args[@]}"

#!/bin/bash
#SBATCH --job-name=mint-agent-dataset-preparation
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

set -euo pipefail

MINT_AGENT_ROOT=${MINT_AGENT_ROOT:?MINT_AGENT_ROOT must be provided by the execution profile}
TASK_CONFIG=${TASK_CONFIG:?TASK_CONFIG must be provided by the execution profile}
RUN_ROOT=${RUN_ROOT:-${MINT_AGENT_ROOT}/runs}
LOG_ROOT=${LOG_ROOT:-${MINT_AGENT_ROOT}/logs}
PYTHON_BIN=${PYTHON_BIN:-python}
OUTPUT_REPORT=${OUTPUT_REPORT:?OUTPUT_REPORT must be provided by the execution profile}

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

if [[ -n "${DATASET_PREPARATION_MODULES:-}" ]]; then
  if [[ "${DATASET_PREPARATION_MODULE_PURGE:-0}" == "1" ]]; then
    module purge
  fi
  IFS=':' read -r -a preparation_modules <<< "${DATASET_PREPARATION_MODULES}"
  for module_name in "${preparation_modules[@]}"; do
    module load "${module_name}"
  done
fi

"${PYTHON_BIN}" -m mint_scout.prepare_dataset \
  --config "${TASK_CONFIG}" \
  --output "${OUTPUT_REPORT}"

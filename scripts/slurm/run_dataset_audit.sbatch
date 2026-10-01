#!/bin/bash
#SBATCH --job-name=mint-agent-dataset-audit
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

set -euo pipefail

MINT_AGENT_ROOT=${MINT_AGENT_ROOT:?MINT_AGENT_ROOT must be provided by the execution profile}
TASK_CONFIG=${TASK_CONFIG:?TASK_CONFIG must be provided by the execution profile}
RUN_ROOT=${RUN_ROOT:-${MINT_AGENT_ROOT}/runs}
LOG_ROOT=${LOG_ROOT:-${MINT_AGENT_ROOT}/logs}
PYTHON_BIN=${PYTHON_BIN:-python}
OUTPUT_JSON=${OUTPUT_JSON:?OUTPUT_JSON must be provided by the execution profile}
OUTPUT_MD=${OUTPUT_MD:-${OUTPUT_JSON%.json}.md}

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

"${PYTHON_BIN}" -m mint_scout.data_audit \
  --config "${TASK_CONFIG}" \
  --output-json "${OUTPUT_JSON}" \
  --output-md "${OUTPUT_MD}"

#!/bin/bash
#SBATCH --job-name=mint-agent-representation-design
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

set -euo pipefail

MINT_AGENT_ROOT=${MINT_AGENT_ROOT:?MINT_AGENT_ROOT must be provided by the execution profile}
TASK_CONFIG=${TASK_CONFIG:?TASK_CONFIG must be provided by the execution profile}
DATA_AUDIT_REPORT=${DATA_AUDIT_REPORT:?DATA_AUDIT_REPORT must be provided}
DESIGN_INPUT_HASH=${DESIGN_INPUT_HASH:?DESIGN_INPUT_HASH must be provided}
RUN_ROOT=${RUN_ROOT:-${MINT_AGENT_ROOT}/runs}
LOG_ROOT=${LOG_ROOT:-${MINT_AGENT_ROOT}/logs}
PYTHON_BIN=${PYTHON_BIN:-python}
SPLIT=${SPLIT:-train}
OFFSET=${OFFSET:-0}
LIMIT=${LIMIT:-all}
REPORT=${REPORT:?REPORT must be provided by the execution profile}

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
  --split "${SPLIT}"
  --offset "${OFFSET}"
  --data-audit-report "${DATA_AUDIT_REPORT}"
  --expected-input-hash "${DESIGN_INPUT_HASH}"
  --output "${REPORT}"
)
if [[ "${LIMIT}" != "all" ]]; then
  args+=(--limit "${LIMIT}")
fi

"${PYTHON_BIN}" -m mint_scout.design_representation "${args[@]}"

#!/bin/bash
#SBATCH --job-name=mint-agent-probe-audit
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

set -euo pipefail

MINT_AGENT_ROOT=${MINT_AGENT_ROOT:?MINT_AGENT_ROOT must be provided by the execution profile}
TASK_CONFIG=${TASK_CONFIG:?TASK_CONFIG must be provided by the execution profile}
SCOUT_CONFIG=${SCOUT_CONFIG:?SCOUT_CONFIG must be provided by the execution profile}
PROBE_SELECTION=${PROBE_SELECTION:?PROBE_SELECTION must be provided}
OUTPUT=${OUTPUT:?OUTPUT must be provided by the execution profile}
RUN_ROOT=${RUN_ROOT:-${MINT_AGENT_ROOT}/runs}
LOG_ROOT=${LOG_ROOT:-${MINT_AGENT_ROOT}/logs}
PYTHON_BIN=${PYTHON_BIN:-python}

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

"${PYTHON_BIN}" -m mint_scout.probe_audit \
  --task-config "${TASK_CONFIG}" \
  --scout-config "${SCOUT_CONFIG}" \
  --probe-selection "${PROBE_SELECTION}" \
  --output "${OUTPUT}"

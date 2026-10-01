#!/bin/bash
#SBATCH --job-name=mint-agent-probe-selection
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

set -euo pipefail

MINT_AGENT_ROOT=${MINT_AGENT_ROOT:?MINT_AGENT_ROOT must be provided by the execution profile}
TASK_CONFIG=${TASK_CONFIG:?TASK_CONFIG must be provided by the execution profile}
SCOUT_CONFIG=${SCOUT_CONFIG:?SCOUT_CONFIG must be provided by the execution profile}
REPRESENTATION_SPEC=${REPRESENTATION_SPEC:?REPRESENTATION_SPEC must be provided}
OUTPUT=${OUTPUT:?OUTPUT must be provided by the execution profile}
RUN_ROOT=${RUN_ROOT:-${MINT_AGENT_ROOT}/runs}
LOG_ROOT=${LOG_ROOT:-${MINT_AGENT_ROOT}/logs}
PYTHON_BIN=${PYTHON_BIN:-python}
FROZEN_SAMPLE_SOURCE=${FROZEN_SAMPLE_SOURCE:-}
SAMPLING_STRATEGY=${SAMPLING_STRATEGY:-}
PROBE_RANDOM_SEED=${PROBE_RANDOM_SEED:-}
PROBE_MAX_SAMPLES=${PROBE_MAX_SAMPLES:-}

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
  --task-config "${TASK_CONFIG}"
  --scout-config "${SCOUT_CONFIG}"
  --representation-spec "${REPRESENTATION_SPEC}"
  --output "${OUTPUT}"
)
[[ -n "${FROZEN_SAMPLE_SOURCE}" ]] && args+=(--frozen-sample-source "${FROZEN_SAMPLE_SOURCE}")
[[ -n "${SAMPLING_STRATEGY}" ]] && args+=(--sampling-strategy "${SAMPLING_STRATEGY}")
[[ -n "${PROBE_RANDOM_SEED}" ]] && args+=(--random-seed "${PROBE_RANDOM_SEED}")
[[ -n "${PROBE_MAX_SAMPLES}" ]] && args+=(--probe-max-samples "${PROBE_MAX_SAMPLES}")

"${PYTHON_BIN}" -m mint_scout.select_casf_probe "${args[@]}"

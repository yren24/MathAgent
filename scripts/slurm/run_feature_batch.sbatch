#!/bin/bash
#SBATCH --job-name=mint-agent-feature-batch
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

set -euo pipefail

MINT_AGENT_ROOT=${MINT_AGENT_ROOT:?MINT_AGENT_ROOT must be provided by the execution profile}
TASK_CONFIG=${TASK_CONFIG:?TASK_CONFIG must be provided by the execution profile}
RUN_ROOT=${RUN_ROOT:-${MINT_AGENT_ROOT}/runs}
CACHE_ROOT=${CACHE_ROOT:-${MINT_AGENT_ROOT}/cache}
LOG_ROOT=${LOG_ROOT:-${MINT_AGENT_ROOT}/logs}
PYTHON_BIN=${PYTHON_BIN:-python}
INVARIANT=${INVARIANT:?INVARIANT must be provided}
SPLIT=${SPLIT:-all}
OFFSET=${OFFSET:-0}
LIMIT=${LIMIT:-all}
SAMPLE_ID_FILE=${SAMPLE_ID_FILE:-}
REPRESENTATION_SPEC=${REPRESENTATION_SPEC:-}
LEGACY_ROOT=${LEGACY_ROOT:-}
MANIFEST=${MANIFEST:-${RUN_ROOT}/feature_batch_${INVARIANT}_${SPLIT}_${SLURM_JOB_ID:-manual}.jsonl}

mkdir -p "${LOG_ROOT}" "${RUN_ROOT}" "${CACHE_ROOT}"
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
  --invariant "${INVARIANT}"
  --split "${SPLIT}"
  --offset "${OFFSET}"
  --manifest "${MANIFEST}"
)
if [[ "${LIMIT}" != "all" ]]; then
  args+=(--limit "${LIMIT}")
fi
if [[ -n "${SAMPLE_ID_FILE}" ]]; then
  args+=(--sample-id-file "${SAMPLE_ID_FILE}")
fi
if [[ -n "${REPRESENTATION_SPEC}" ]]; then
  args+=(--representation-spec "${REPRESENTATION_SPEC}")
fi
if [[ -n "${LEGACY_ROOT}" ]]; then
  args+=(--legacy-root "${LEGACY_ROOT}")
fi

"${PYTHON_BIN}" -m mint_scout.compute_feature_batch "${args[@]}"

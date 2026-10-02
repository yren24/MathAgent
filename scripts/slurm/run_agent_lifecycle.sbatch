#!/bin/bash
#SBATCH --job-name=mint-agent-lifecycle
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

set -euo pipefail

MINT_AGENT_ROOT=${MINT_AGENT_ROOT:?MINT_AGENT_ROOT must be provided}
PYTHON_BIN=${PYTHON_BIN:-python}
RUN_ROOT=${RUN_ROOT:?RUN_ROOT must be provided}
LOG_ROOT=${LOG_ROOT:-$(dirname "${RUN_ROOT}")/logs}
REGISTRY=${REGISTRY:?REGISTRY must be provided}
TASK_CONFIG=${TASK_CONFIG:?TASK_CONFIG must be provided}
EXECUTION_PROFILE=${EXECUTION_PROFILE:?EXECUTION_PROFILE must be provided}
DATASET_ID=${DATASET_ID:?DATASET_ID must be provided}
INVARIANTS=${INVARIANTS:?INVARIANTS must be provided}
USER_TARGET=${USER_TARGET:-}
EVIDENCE_SCOPE=${EVIDENCE_SCOPE:-full_train}
REPRESENTATION_SPEC=${REPRESENTATION_SPEC:-}
SAMPLE_ID_FILE=${SAMPLE_ID_FILE:-}
LIFECYCLE=${LIFECYCLE:-${RUN_ROOT}/agent_lifecycle_${DATASET_ID}.json}
LIFECYCLE_ID=${LIFECYCLE_ID:-${DATASET_ID}-gbt-lifecycle}
CONTINUATION_SCRIPT=${CONTINUATION_SCRIPT:-scripts/slurm/run_agent_lifecycle.sbatch}
EXECUTE=${EXECUTE:-0}
AUTO_CONTINUE=${AUTO_CONTINUE:-0}
AUTO_RESUME_FAILED=${AUTO_RESUME_FAILED:-0}
MAX_ATTEMPTS=${MAX_ATTEMPTS:-2}
STOP_AFTER_STAGE=${STOP_AFTER_STAGE:-}
FEATURE_DIAGNOSTIC_TOP_K=${FEATURE_DIAGNOSTIC_TOP_K:-10}
QC_CONFIG=${QC_CONFIG:-configs/scout/v1.yaml}
AUDIT_CONFIG=${AUDIT_CONFIG:-configs/scout/v1.yaml}
SCOUT_CONFIG=${SCOUT_CONFIG:-configs/scout/v1.yaml}
GBT_CONFIG=${GBT_CONFIG:-configs/gbt/plbind_fixed_gbt.yaml}
LLM_SCIENTIFIC_MODE=${LLM_SCIENTIFIC_MODE:-disabled}
LLM_MODEL=${LLM_MODEL:-}
LLM_API_KEY_ENV=${LLM_API_KEY_ENV:-OPENAI_API_KEY}
LLM_ENV_FILE=${LLM_ENV_FILE:-}
LLM_CACHE_DIR=${LLM_CACHE_DIR:-}
LLM_MAX_CANDIDATES=${LLM_MAX_CANDIDATES:-10}
LLM_TIMEOUT_SECONDS=${LLM_TIMEOUT_SECONDS:-180}

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

advance_args=(
  --state "${LIFECYCLE}"
  --continuation-script "${CONTINUATION_SCRIPT}"
  --project-root "${MINT_AGENT_ROOT}"
  --max-attempts "${MAX_ATTEMPTS}"
)
[[ "${EXECUTE}" == "1" ]] && advance_args+=(--execute)
[[ "${AUTO_CONTINUE}" == "1" ]] && advance_args+=(--auto-continue)
[[ "${AUTO_RESUME_FAILED}" == "1" ]] && advance_args+=(--resume-failed)
[[ -n "${STOP_AFTER_STAGE}" ]] && advance_args+=(--stop-after-stage "${STOP_AFTER_STAGE}")

if [[ -f "${LIFECYCLE}" ]]; then
  "${PYTHON_BIN}" -m mint_scout.run_agent_lifecycle advance "${advance_args[@]}"
  exit $?
fi

start_args=(
  start
  "${advance_args[@]}"
  --task-config "${TASK_CONFIG}"
  --registry "${REGISTRY}"
  --dataset-id "${DATASET_ID}"
  --execution-profile "${EXECUTION_PROFILE}"
  --evidence-scope "${EVIDENCE_SCOPE}"
  --lifecycle-id "${LIFECYCLE_ID}"
  --feature-diagnostic-top-k "${FEATURE_DIAGNOSTIC_TOP_K}"
  --qc-config "${QC_CONFIG}"
  --audit-config "${AUDIT_CONFIG}"
  --scout-config "${SCOUT_CONFIG}"
  --gbt-config "${GBT_CONFIG}"
  --llm-scientific-mode "${LLM_SCIENTIFIC_MODE}"
  --llm-api-key-env "${LLM_API_KEY_ENV}"
  --llm-max-candidates "${LLM_MAX_CANDIDATES}"
  --llm-timeout-seconds "${LLM_TIMEOUT_SECONDS}"
)
normalized_invariants=${INVARIANTS//,/ }
for invariant in ${normalized_invariants}; do
  start_args+=(--invariant "${invariant}")
done
[[ -n "${REPRESENTATION_SPEC}" ]] && start_args+=(--representation-spec "${REPRESENTATION_SPEC}")
[[ -n "${SAMPLE_ID_FILE}" ]] && start_args+=(--sample-id-file "${SAMPLE_ID_FILE}")
[[ -n "${USER_TARGET}" ]] && start_args+=(--user-target "${USER_TARGET}")
[[ -n "${LLM_MODEL}" ]] && start_args+=(--llm-model "${LLM_MODEL}")
[[ -n "${LLM_ENV_FILE}" ]] && start_args+=(--llm-env-file "${LLM_ENV_FILE}")
[[ -n "${LLM_CACHE_DIR}" ]] && start_args+=(--llm-cache-dir "${LLM_CACHE_DIR}")

"${PYTHON_BIN}" -m mint_scout.run_agent_lifecycle "${start_args[@]}"

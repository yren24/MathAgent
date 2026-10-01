#!/bin/bash
#SBATCH --job-name=mint-agent-user-start
#SBATCH --partition=batch
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2gb
#SBATCH --time=00:15:00
#SBATCH --output=/path/to/workdir/mathagent/logs/%x-%j.out
#SBATCH --error=/path/to/workdir/mathagent/logs/%x-%j.err

set -euo pipefail

MINT_AGENT_ROOT=${MINT_AGENT_ROOT:-/path/to/MathAgent}
PYTHON_BIN=${PYTHON_BIN:-${MINT_AGENT_ROOT}/.venv-langgraph/bin/python}
EXECUTE=${EXECUTE:-0}
OPENAI_ENV_FILE=${OPENAI_ENV_FILE:-~/.config/mathagent/openai.env}
MANIFEST_PROFILE_MAX_ROWS=${MANIFEST_PROFILE_MAX_ROWS:-200}
LLM_SCIENTIFIC_MODE=${LLM_SCIENTIFIC_MODE:-disabled}
LLM_SCIENTIFIC_MODEL=${LLM_SCIENTIFIC_MODEL:-${OPENAI_MODEL:-}}
LLM_SCIENTIFIC_ENV_FILE=${LLM_SCIENTIFIC_ENV_FILE:-${OPENAI_ENV_FILE}}
LLM_SCIENTIFIC_CACHE_DIR=${LLM_SCIENTIFIC_CACHE_DIR:-}
LLM_SCIENTIFIC_MAX_CANDIDATES=${LLM_SCIENTIFIC_MAX_CANDIDATES:-10}
LLM_SCIENTIFIC_TIMEOUT_SECONDS=${LLM_SCIENTIFIC_TIMEOUT_SECONDS:-180}

if [[ -z "${USER_REQUEST:-}" && -z "${USER_PROMPT_FILE:-}" ]]; then
  echo "Set USER_REQUEST or USER_PROMPT_FILE" >&2
  exit 2
fi
if [[ -n "${USER_REQUEST:-}" && -n "${USER_PROMPT_FILE:-}" ]]; then
  echo "Set only one of USER_REQUEST or USER_PROMPT_FILE" >&2
  exit 2
fi

cd "${MINT_AGENT_ROOT}"
source scripts/sapelo2/load_modules.sh
if [[ -f "${OPENAI_ENV_FILE}" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "${OPENAI_ENV_FILE}"
  set +a
fi
unset EBPYTHONPREFIXES
export PYTHONPATH=${MINT_AGENT_ROOT}/src

args=()
if [[ -n "${USER_REQUEST:-}" ]]; then
  args+=(--request "${USER_REQUEST}")
else
  args+=(--prompt-file "${USER_PROMPT_FILE}")
  args+=(--model "${OPENAI_MODEL:?OPENAI_MODEL is required for USER_PROMPT_FILE}")
  if [[ -n "${EXECUTION_PROFILE_OVERRIDE:-}" ]]; then
    args+=(--execution-profile "${EXECUTION_PROFILE_OVERRIDE}")
  fi
  if [[ -n "${REPRESENTATION_PROFILE_OVERRIDE:-}" ]]; then
    args+=(--representation-profile "${REPRESENTATION_PROFILE_OVERRIDE}")
  fi
  if [[ -n "${MANIFEST_PATH_OVERRIDE:-}" ]]; then
    args+=(--manifest-path "${MANIFEST_PATH_OVERRIDE}")
  fi
  args+=(--manifest-profile-max-rows "${MANIFEST_PROFILE_MAX_ROWS}")
  args+=(--llm-scientific-mode "${LLM_SCIENTIFIC_MODE}")
  args+=(--llm-scientific-max-candidates "${LLM_SCIENTIFIC_MAX_CANDIDATES}")
  args+=(--llm-scientific-timeout-seconds "${LLM_SCIENTIFIC_TIMEOUT_SECONDS}")
  if [[ -n "${LLM_SCIENTIFIC_MODEL}" ]]; then
    args+=(--llm-scientific-model "${LLM_SCIENTIFIC_MODEL}")
  fi
  if [[ -n "${LLM_SCIENTIFIC_ENV_FILE}" ]]; then
    args+=(--llm-scientific-env-file "${LLM_SCIENTIFIC_ENV_FILE}")
  fi
  if [[ -n "${LLM_SCIENTIFIC_CACHE_DIR}" ]]; then
    args+=(--llm-scientific-cache-dir "${LLM_SCIENTIFIC_CACHE_DIR}")
  fi
fi
if [[ -n "${OUTPUT_DIR:-}" ]]; then
  args+=(--output-dir "${OUTPUT_DIR}")
fi
if [[ "${EXECUTE}" == "1" ]]; then
  args+=(--execute)
fi

"${PYTHON_BIN}" -m mint_scout.user_intake "${args[@]}"

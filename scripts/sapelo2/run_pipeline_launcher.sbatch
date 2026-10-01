#!/bin/bash
#SBATCH --job-name=mint-agent-pipeline-launch
#SBATCH --partition=batch
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2gb
#SBATCH --time=00:10:00
#SBATCH --output=/path/to/workdir/mathagent/logs/%x-%j.out
#SBATCH --error=/path/to/workdir/mathagent/logs/%x-%j.err

set -euo pipefail

MINT_AGENT_ROOT=${MINT_AGENT_ROOT:-/path/to/MathAgent}
PYTHON_BIN=${PYTHON_BIN:-${MINT_AGENT_ROOT}/.venv-langgraph/bin/python}
PIPELINE_CONFIG=${PIPELINE_CONFIG:?PIPELINE_CONFIG must be provided}
PIPELINE_COMMAND=${PIPELINE_COMMAND:-submit}
EXECUTE=${EXECUTE:-0}

case "${PIPELINE_COMMAND}" in
  plan|submit|status|resume|report) ;;
  *)
    echo "Unsupported PIPELINE_COMMAND: ${PIPELINE_COMMAND}" >&2
    exit 2
    ;;
esac

mkdir -p /path/to/workdir/mathagent/logs
cd "${MINT_AGENT_ROOT}"
source scripts/sapelo2/load_modules.sh
unset EBPYTHONPREFIXES
export PYTHONPATH=${MINT_AGENT_ROOT}/src

args=("${PIPELINE_COMMAND}" --config "${PIPELINE_CONFIG}")
if [[ "${EXECUTE}" == "1" ]]; then
  case "${PIPELINE_COMMAND}" in
    submit|resume) args+=(--execute) ;;
  esac
fi

"${PYTHON_BIN}" -m mint_scout.run_pipeline "${args[@]}"

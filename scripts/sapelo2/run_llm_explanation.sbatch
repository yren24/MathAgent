#!/bin/bash
#SBATCH --job-name=mint-agent-explain
#SBATCH --partition=batch
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2gb
#SBATCH --time=00:10:00
#SBATCH --output=/path/to/workdir/mathagent/logs/%x-%j.out
#SBATCH --error=/path/to/workdir/mathagent/logs/%x-%j.err

set -euo pipefail

MINT_AGENT_ROOT=${MINT_AGENT_ROOT:-/path/to/MathAgent}
PYTHON_BIN=${PYTHON_BIN:-python}
SUMMARY_REPORT=${SUMMARY_REPORT:?SUMMARY_REPORT must be provided}
OPENAI_MODEL=${OPENAI_MODEL:?OPENAI_MODEL must be provided}
OPENAI_API_KEY=${OPENAI_API_KEY:?OPENAI_API_KEY must be set in the environment}
EXPLANATION_LANGUAGE=${EXPLANATION_LANGUAGE:-English}
EXPLANATION_OUTPUT=${EXPLANATION_OUTPUT:-}

mkdir -p /path/to/workdir/mathagent/logs
cd "${MINT_AGENT_ROOT}"
source scripts/sapelo2/load_modules.sh
unset EBPYTHONPREFIXES
export PYTHONPATH=${MINT_AGENT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}

args=(
  --summary "${SUMMARY_REPORT}"
  --model "${OPENAI_MODEL}"
  --language "${EXPLANATION_LANGUAGE}"
)
if [[ -n "${EXPLANATION_OUTPUT}" ]]; then
  args+=(--output "${EXPLANATION_OUTPUT}")
fi

"${PYTHON_BIN}" -m mint_scout.agent.llm_explanation "${args[@]}"

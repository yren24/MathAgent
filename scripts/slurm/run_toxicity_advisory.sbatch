#!/bin/bash
#SBATCH --job-name=mint-tox-advice
#SBATCH --partition=batch
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/path/to/workdir/mathagent/logs/toxicity-advisory-%j.out
#SBATCH --error=/path/to/workdir/mathagent/logs/toxicity-advisory-%j.err

set -euo pipefail

MINT_AGENT_ROOT=${MINT_AGENT_ROOT:-/path/to/MathAgent}
BASE_DESIGN_ROOT=${BASE_DESIGN_ROOT:-/path/to/workdir/mathagent/toxicity/design-v1}
OUTPUT_ROOT=${OUTPUT_ROOT:-/path/to/workdir/mathagent/toxicity/design-v1-advisory}
DATASET_ID=${DATASET_ID:-LD50}
LLM_ENV_FILE=${LLM_ENV_FILE:-~/.config/mathagent/openai.env}
LLM_CACHE_DIR=${LLM_CACHE_DIR:-/path/to/workdir/mathagent/memory/llm}
LLM_MODEL=${LLM_MODEL:-}
AGENT_ENV=${AGENT_ENV:-/path/to/MathAgent/.venv-langgraph}
PYTHON_BIN=${PYTHON_BIN:-${AGENT_ENV}/bin/python}
GUDHI_ROOT=${GUDHI_ROOT:-/path/to/workdir/mathagent/deps/gudhi-3.13.0}

mkdir -p /path/to/workdir/mathagent/logs "${OUTPUT_ROOT}"
cd "${MINT_AGENT_ROOT}"
source scripts/sapelo2/load_modules.sh
unset EBPYTHONPREFIXES
export PYTHONPATH="${MINT_AGENT_ROOT}/src:${GUDHI_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1

if [[ ! -f "${LLM_ENV_FILE}" ]]; then
  echo "LLM environment file not found: ${LLM_ENV_FILE}" >&2
  exit 2
fi
set -a
source "${LLM_ENV_FILE}"
set +a
if [[ -n "${LLM_MODEL}" ]]; then
  export OPENAI_MODEL="${LLM_MODEL}"
fi

"${PYTHON_BIN}" -m pytest -q \
  tests/test_toxicity_legacy.py \
  tests/test_toxicity_design.py \
  tests/test_toxicity_advisory.py

"${PYTHON_BIN}" -m mint_scout.toxicity.advisory \
  --design-report "${BASE_DESIGN_ROOT}/toxicity_representation_design.json" \
  --artifact "${OUTPUT_ROOT}/toxicity_llm_advisory.json" \
  --proposals "${OUTPUT_ROOT}/toxicity_llm_proposals.json" \
  --cache-dir "${LLM_CACHE_DIR}"

"${PYTHON_BIN}" -m mint_scout.toxicity.design \
  --manifest "${BASE_DESIGN_ROOT}/ld50_manifest.csv" \
  --dataset-id "${DATASET_ID}" \
  --primary-metric "${PRIMARY_METRIC:-PCC2}" \
  --llm-proposals "${OUTPUT_ROOT}/toxicity_llm_proposals.json" \
  --output-dir "${OUTPUT_ROOT}"

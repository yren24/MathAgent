#!/bin/bash
#SBATCH --job-name=mint-tox-workflow
#SBATCH --partition=batch
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:20:00
#SBATCH --output=/path/to/workdir/mathagent/logs/toxicity-workflow-%j.out
#SBATCH --error=/path/to/workdir/mathagent/logs/toxicity-workflow-%j.err

set -euo pipefail

MINT_AGENT_ROOT=${MINT_AGENT_ROOT:-/path/to/MathAgent}
DESIGN_ROOT=${DESIGN_ROOT:-/path/to/workdir/mathagent/toxicity/design-v1}
ADVISORY_ROOT=${ADVISORY_ROOT:-/path/to/workdir/mathagent/toxicity/design-v1-advisory}
PROBE_ROOT=${PROBE_ROOT:-/path/to/workdir/mathagent/toxicity/probe-v1}
FEATURE_PLAN_ROOT=${FEATURE_PLAN_ROOT:-${PROBE_ROOT}/features}
FEATURE_ROOT=${FEATURE_ROOT:-/path/to/workdir/mathagent/toxicity/feature-cache}
REPAIR_ROOT=${REPAIR_ROOT:-${PROBE_ROOT}/repair-v1}
FINAL_ROOT=${FINAL_ROOT:-/path/to/workdir/mathagent/toxicity/final-v1}
LEGACY_ROOT=${LEGACY_ROOT:-/path/to/workdir/mathagent/external/toxicity-phase0/legacy}
MANIFEST_PATH=${MANIFEST_PATH:-}
MOLECULE_DIRS=${MOLECULE_DIRS:-}
DATASET_ID=${DATASET_ID:-LD50}
GBT_CONFIG=${GBT_CONFIG:-${MINT_AGENT_ROOT}/configs/gbt/toxicity_probe_gbt.yaml}
PRIMARY_METRIC=${PRIMARY_METRIC:-PCC2}
LLM_ENABLED=${LLM_ENABLED:-0}
LLM_MODEL=${LLM_MODEL:-}
LLM_ENV_FILE=${LLM_ENV_FILE:-~/.config/mathagent/openai.env}
LLM_CACHE_DIR=${LLM_CACHE_DIR:-/path/to/workdir/mathagent/memory/llm}
WORKFLOW_JOBS_PATH=${WORKFLOW_JOBS_PATH:-${FINAL_ROOT}/../toxicity.jobs.json}
AGENT_ENV=${AGENT_ENV:-/path/to/MathAgent/.venv-langgraph}
PYTHON_BIN=${PYTHON_BIN:-${AGENT_ENV}/bin/python}
GUDHI_ROOT=${GUDHI_ROOT:-/path/to/workdir/mathagent/deps/gudhi-3.13.0}

mkdir -p /path/to/workdir/mathagent/logs "${DESIGN_ROOT}" "${PROBE_ROOT}" "${FINAL_ROOT}"
cd "${MINT_AGENT_ROOT}"
source scripts/sapelo2/load_modules.sh
unset EBPYTHONPREFIXES
export PYTHONPATH="${MINT_AGENT_ROOT}/src:${GUDHI_ROOT}"
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1

"${PYTHON_BIN}" -m mint_scout.toxicity.agent_graph \
  --execute \
  --output "${WORKFLOW_JOBS_PATH}.graph.json"

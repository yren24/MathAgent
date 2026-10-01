#!/bin/bash
#SBATCH --job-name=mint-tox-final-plan
#SBATCH --partition=batch
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --output=/path/to/workdir/mathagent/logs/toxicity-final-plan-%j.out
#SBATCH --error=/path/to/workdir/mathagent/logs/toxicity-final-plan-%j.err

set -euo pipefail

MINT_AGENT_ROOT=${MINT_AGENT_ROOT:-/path/to/MathAgent}
PROBE_ROOT=${PROBE_ROOT:-/path/to/workdir/mathagent/toxicity/probe-v1}
REPAIR_ROOT=${REPAIR_ROOT:-${PROBE_ROOT}/repair-v1}
FINAL_ROOT=${FINAL_ROOT:-/path/to/workdir/mathagent/toxicity/final-v1}
AGENT_ENV=${AGENT_ENV:-/path/to/MathAgent/.venv-langgraph}
PYTHON_BIN=${PYTHON_BIN:-${AGENT_ENV}/bin/python}

mkdir -p /path/to/workdir/mathagent/logs "${FINAL_ROOT}"
cd "${MINT_AGENT_ROOT}"
source scripts/sapelo2/load_modules.sh
unset EBPYTHONPREFIXES
export PYTHONPATH="${MINT_AGENT_ROOT}/src"

"${PYTHON_BIN}" -m mint_scout.toxicity.final_plan \
  --scout-report "${REPAIR_ROOT}/gbt/toxicity_probe_scout.json" \
  --feature-plan "${REPAIR_ROOT}/toxicity_repair_feature_plan.json" \
  --output-root "${FINAL_ROOT}" \
  --output "${FINAL_ROOT}/toxicity_final_plan.json"

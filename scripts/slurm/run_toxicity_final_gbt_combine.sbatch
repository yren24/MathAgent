#!/bin/bash
#SBATCH --job-name=mint-tox-final-combine
#SBATCH --partition=batch
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/path/to/workdir/mathagent/logs/toxicity-final-combine-%j.out
#SBATCH --error=/path/to/workdir/mathagent/logs/toxicity-final-combine-%j.err

set -euo pipefail

MINT_AGENT_ROOT=${MINT_AGENT_ROOT:-/path/to/MathAgent}
FINAL_ROOT=${FINAL_ROOT:-/path/to/workdir/mathagent/toxicity/final-v1}
AGENT_ENV=${AGENT_ENV:-/path/to/MathAgent/.venv-langgraph}
PYTHON_BIN=${PYTHON_BIN:-${AGENT_ENV}/bin/python}

mkdir -p /path/to/workdir/mathagent/logs "${FINAL_ROOT}/gbt"
cd "${MINT_AGENT_ROOT}"
source scripts/sapelo2/load_modules.sh
unset EBPYTHONPREFIXES
export PYTHONPATH="${MINT_AGENT_ROOT}/src"

"${PYTHON_BIN}" -m mint_scout.toxicity.final_gbt combine \
  --plan "${FINAL_ROOT}/toxicity_final_plan.json" \
  --method-report-dir "${FINAL_ROOT}/gbt/methods" \
  --n-bootstrap 1000 \
  --bootstrap-seed 2026 \
  --output "${FINAL_ROOT}/gbt/toxicity_final_test_report.json"

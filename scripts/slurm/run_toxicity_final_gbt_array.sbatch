#!/bin/bash
#SBATCH --job-name=mint-tox-final-gbt
#SBATCH --partition=batch
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=/path/to/workdir/mathagent/logs/toxicity-final-gbt-%A_%a.out
#SBATCH --error=/path/to/workdir/mathagent/logs/toxicity-final-gbt-%A_%a.err

set -euo pipefail

MINT_AGENT_ROOT=${MINT_AGENT_ROOT:-/path/to/MathAgent}
FINAL_ROOT=${FINAL_ROOT:-/path/to/workdir/mathagent/toxicity/final-v1}
GBT_CONFIG=${GBT_CONFIG:-${MINT_AGENT_ROOT}/configs/gbt/toxicity_probe_gbt.yaml}
AGENT_ENV=${AGENT_ENV:-/path/to/MathAgent/.venv-langgraph}
PYTHON_BIN=${PYTHON_BIN:-${AGENT_ENV}/bin/python}

mkdir -p /path/to/workdir/mathagent/logs "${FINAL_ROOT}/gbt/methods"
cd "${MINT_AGENT_ROOT}"
source scripts/sapelo2/load_modules.sh
unset EBPYTHONPREFIXES
export PYTHONPATH="${MINT_AGENT_ROOT}/src"

"${PYTHON_BIN}" -m mint_scout.toxicity.final_gbt method \
  --plan "${FINAL_ROOT}/toxicity_final_plan.json" \
  --gbt-config "${GBT_CONFIG}" \
  --method-index "${SLURM_ARRAY_TASK_ID}" \
  --output-dir "${FINAL_ROOT}/gbt/methods"

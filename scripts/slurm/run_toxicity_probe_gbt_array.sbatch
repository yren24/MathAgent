#!/bin/bash
#SBATCH --job-name=mint-tox-gbt
#SBATCH --partition=batch
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --time=12:00:00
#SBATCH --output=/path/to/workdir/mathagent/logs/toxicity-gbt-%A_%a.out
#SBATCH --error=/path/to/workdir/mathagent/logs/toxicity-gbt-%A_%a.err

set -euo pipefail

MINT_AGENT_ROOT=${MINT_AGENT_ROOT:-/path/to/MathAgent}
PROBE_ROOT=${PROBE_ROOT:-/path/to/workdir/mathagent/toxicity/probe-v1}
REPAIR_ROOT=${REPAIR_ROOT:-${PROBE_ROOT}/repair-v1}
GBT_CONFIG=${GBT_CONFIG:-${MINT_AGENT_ROOT}/configs/gbt/toxicity_probe_gbt.yaml}
GBT_ROOT=${GBT_ROOT:-${REPAIR_ROOT}/gbt}
AGENT_ENV=${AGENT_ENV:-/path/to/MathAgent/.venv-langgraph}
PYTHON_BIN=${PYTHON_BIN:-${AGENT_ENV}/bin/python}

mkdir -p /path/to/workdir/mathagent/logs "${GBT_ROOT}/oof"
cd "${MINT_AGENT_ROOT}"
source scripts/sapelo2/load_modules.sh
unset EBPYTHONPREFIXES
export PYTHONPATH="${MINT_AGENT_ROOT}/src"

"${PYTHON_BIN}" -m mint_scout.toxicity.probe_gbt oof \
  --feature-plan "${REPAIR_ROOT}/toxicity_repair_feature_plan.json" \
  --feature-qc "${REPAIR_ROOT}/toxicity_repair_feature_qc.json" \
  --probe-selection "${PROBE_ROOT}/toxicity_probe_selection.json" \
  --gbt-config "${GBT_CONFIG}" \
  --task-index "${SLURM_ARRAY_TASK_ID}" \
  --output-dir "${GBT_ROOT}/oof"

#!/bin/bash
#SBATCH --job-name=mint-tox-repair
#SBATCH --partition=batch
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#SBATCH --output=/path/to/workdir/mathagent/logs/toxicity-repair-%j.out
#SBATCH --error=/path/to/workdir/mathagent/logs/toxicity-repair-%j.err

set -euo pipefail

MINT_AGENT_ROOT=${MINT_AGENT_ROOT:-/path/to/MathAgent}
PROBE_ROOT=${PROBE_ROOT:-/path/to/workdir/mathagent/toxicity/probe-v1}
FEATURE_ROOT=${FEATURE_ROOT:-${PROBE_ROOT}/features}
REPAIR_ROOT=${REPAIR_ROOT:-${PROBE_ROOT}/repair-v1}
AGENT_ENV=${AGENT_ENV:-/path/to/MathAgent/.venv-langgraph}
PYTHON_BIN=${PYTHON_BIN:-${AGENT_ENV}/bin/python}

mkdir -p /path/to/workdir/mathagent/logs "${REPAIR_ROOT}"
cd "${MINT_AGENT_ROOT}"
source scripts/sapelo2/load_modules.sh
unset EBPYTHONPREFIXES
export PYTHONPATH="${MINT_AGENT_ROOT}/src"
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1

"${PYTHON_BIN}" -m pytest -q tests/test_toxicity_filtration_repair.py

"${PYTHON_BIN}" -m mint_scout.toxicity.filtration_repair \
  --representation-selection "${PROBE_ROOT}/toxicity_representation_selection.json" \
  --feature-plan "${FEATURE_ROOT}/toxicity_feature_plan.json" \
  --feature-qc "${FEATURE_ROOT}/toxicity_feature_qc.json" \
  --output-dir "${REPAIR_ROOT}"

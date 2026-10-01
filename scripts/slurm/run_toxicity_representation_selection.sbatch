#!/bin/bash
#SBATCH --job-name=mint-tox-repr
#SBATCH --partition=batch
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#SBATCH --output=/path/to/workdir/mathagent/logs/toxicity-representation-%j.out
#SBATCH --error=/path/to/workdir/mathagent/logs/toxicity-representation-%j.err

set -euo pipefail

MINT_AGENT_ROOT=${MINT_AGENT_ROOT:-/path/to/MathAgent}
DESIGN_ROOT=${DESIGN_ROOT:-/path/to/workdir/mathagent/toxicity/design-v1-advisory}
PROBE_ROOT=${PROBE_ROOT:-/path/to/workdir/mathagent/toxicity/probe-v1}
FEATURE_ROOT=${FEATURE_ROOT:-${PROBE_ROOT}/features}
AGENT_ENV=${AGENT_ENV:-/path/to/MathAgent/.venv-langgraph}
PYTHON_BIN=${PYTHON_BIN:-${AGENT_ENV}/bin/python}

mkdir -p /path/to/workdir/mathagent/logs "${PROBE_ROOT}"
cd "${MINT_AGENT_ROOT}"
source scripts/sapelo2/load_modules.sh
unset EBPYTHONPREFIXES
export PYTHONPATH="${MINT_AGENT_ROOT}/src"
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1

"${PYTHON_BIN}" -m pytest -q tests/test_toxicity_representation_selection.py

"${PYTHON_BIN}" -m mint_scout.toxicity.representation_selection \
  --design-report "${DESIGN_ROOT}/toxicity_representation_design.json" \
  --representation-dir "${DESIGN_ROOT}/representations" \
  --feature-plan "${FEATURE_ROOT}/toxicity_feature_plan.json" \
  --feature-qc "${FEATURE_ROOT}/toxicity_feature_qc.json" \
  --probe-selection "${PROBE_ROOT}/toxicity_probe_selection.json" \
  --output "${PROBE_ROOT}/toxicity_representation_selection.json"

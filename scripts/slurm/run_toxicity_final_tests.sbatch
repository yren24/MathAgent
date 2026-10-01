#!/bin/bash
#SBATCH --job-name=mint-tox-final-test
#SBATCH --partition=batch
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=/path/to/workdir/mathagent/logs/toxicity-final-test-%j.out
#SBATCH --error=/path/to/workdir/mathagent/logs/toxicity-final-test-%j.err

set -euo pipefail

MINT_AGENT_ROOT=${MINT_AGENT_ROOT:-/path/to/MathAgent}
AGENT_ENV=${AGENT_ENV:-/path/to/MathAgent/.venv-langgraph}
PYTHON_BIN=${PYTHON_BIN:-${AGENT_ENV}/bin/python}

mkdir -p /path/to/workdir/mathagent/logs
cd "${MINT_AGENT_ROOT}"
source scripts/sapelo2/load_modules.sh
unset EBPYTHONPREFIXES
export PYTHONPATH="${MINT_AGENT_ROOT}/src"
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1

"${PYTHON_BIN}" -m pytest -q \
  tests/test_toxicity_final.py \
  tests/test_toxicity_probe_gbt.py \
  tests/test_toxicity_filtration_repair.py

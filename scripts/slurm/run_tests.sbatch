#!/bin/bash
#SBATCH --job-name=mint-agent-tests
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8gb
#SBATCH --time=00:30:00

set -euo pipefail

MINT_AGENT_ROOT=${MINT_AGENT_ROOT:-/path/to/MathAgent}
PYTHON_BIN=${PYTHON_BIN:-python}

cd "${MINT_AGENT_ROOT}"
export PYTHONPATH=${MINT_AGENT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}

if [[ -n "${MINT_MODULE_SETUP:-}" ]]; then
  source "${MINT_MODULE_SETUP}"
else
  source scripts/sapelo2/load_modules.sh
fi

"${PYTHON_BIN}" -m pytest

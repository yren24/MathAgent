#!/bin/bash
#SBATCH --job-name=mint-agent-smoke
#SBATCH --partition=batch
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8gb
#SBATCH --time=00:30:00
#SBATCH --output=/path/to/workdir/mathagent/logs/%x-%j.out
#SBATCH --error=/path/to/workdir/mathagent/logs/%x-%j.err

set -euo pipefail

export MINT_AGENT_ROOT=${MINT_AGENT_ROOT:-/path/to/MathAgent}
export PYTHON_BIN=${PYTHON_BIN:-python}
export PYTHONPATH=${MINT_AGENT_ROOT}/src

mkdir -p /path/to/workdir/mathagent/logs
cd "${MINT_AGENT_ROOT}"
source scripts/sapelo2/load_modules.sh

"${PYTHON_BIN}" -m pytest

if [[ "${TEST_ONLY:-0}" == "1" ]]; then
  exit 0
fi

"${PYTHON_BIN}" -m mint_scout.data_audit \
  --config configs/tasks/casf2016_sapelo2.yaml \
  --allow-empty \
  --output-json /path/to/workdir/mathagent/runs/smoke_element_inventory.json \
  --output-md /path/to/workdir/mathagent/runs/smoke_element_inventory.md
"${PYTHON_BIN}" -m mint_scout.smoke_test --config configs/tasks/casf2016_sapelo2.yaml --invariants PH PL --dry-run
"${PYTHON_BIN}" -m mint_scout.feasibility --config configs/tasks/casf2016_sapelo2.yaml --dry-run

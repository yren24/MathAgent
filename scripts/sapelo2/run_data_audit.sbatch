#!/bin/bash
#SBATCH --job-name=mint-agent-data-audit
#SBATCH --partition=batch
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8gb
#SBATCH --time=01:00:00
#SBATCH --output=/path/to/workdir/mathagent/logs/%x-%j.out
#SBATCH --error=/path/to/workdir/mathagent/logs/%x-%j.err

set -euo pipefail

export MINT_AGENT_ROOT=${MINT_AGENT_ROOT:-/path/to/MathAgent}
export PYTHON_BIN=${PYTHON_BIN:-python}
export PYTHONPATH=${MINT_AGENT_ROOT}/src

mkdir -p /path/to/workdir/mathagent/logs /path/to/workdir/mathagent/runs
cd "${MINT_AGENT_ROOT}"
source scripts/sapelo2/load_modules.sh

"${PYTHON_BIN}" -m mint_scout.data_audit \
  --config configs/tasks/casf2016_sapelo2.yaml \
  --output-json /path/to/workdir/mathagent/runs/casf2016_element_inventory.json \
  --output-md /path/to/workdir/mathagent/runs/casf2016_element_inventory.md

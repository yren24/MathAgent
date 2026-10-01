#!/bin/bash
#SBATCH --job-name=mint-scout-data-audit
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --chdir=/path/to/MathAgent
#SBATCH --output=/path/to/MathAgent/logs/data-audit-%j.out

set -euo pipefail

mkdir -p logs docs

export MINT_SCOUT_ROOT=/path/to/MathAgent
export PYTHON_BIN=python
export LD_LIBRARY_PATH=/path/to/python/lib:${LD_LIBRARY_PATH:-}
export PYTHONPATH=${MINT_SCOUT_ROOT}/src

"${PYTHON_BIN}" -m mint_scout.data_audit \
  --config configs/tasks/casf2016_hpcc.yaml \
  --output-json docs/ELEMENT_INVENTORY_HPCC.json \
  --output-md docs/ELEMENT_INVENTORY_HPCC.md

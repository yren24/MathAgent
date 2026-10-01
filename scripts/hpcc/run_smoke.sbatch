#!/bin/bash
#SBATCH --job-name=mint-scout-smoke
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --chdir=/path/to/MathAgent
#SBATCH --output=/path/to/MathAgent/logs/smoke-%j.out

set -euo pipefail

mkdir -p logs

export MINT_SCOUT_ROOT=/path/to/MathAgent
export PYTHON_BIN=python
export LD_LIBRARY_PATH=/path/to/python/lib:${LD_LIBRARY_PATH:-}
export PYTHONPATH=${MINT_SCOUT_ROOT}/src

"${PYTHON_BIN}" -m pytest tests
"${PYTHON_BIN}" -m mint_scout.data_audit \
  --config configs/tasks/casf2016_hpcc.yaml \
  --max-files 10 \
  --output-json "logs/element-inventory-smoke-${SLURM_JOB_ID}.json" \
  --output-md "logs/element-inventory-smoke-${SLURM_JOB_ID}.md"
"${PYTHON_BIN}" -m mint_scout.smoke_test --config configs/tasks/casf2016_hpcc.yaml --invariants PH PL --dry-run
"${PYTHON_BIN}" -m mint_scout.feasibility --config configs/tasks/casf2016_hpcc.yaml --dry-run

#!/bin/bash
#SBATCH --job-name=mint-scout-audit
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --chdir=/path/to/MathAgent
#SBATCH --output=/path/to/MathAgent/logs/audit-%j.out

set -euo pipefail

mkdir -p logs

export MINT_SCOUT_ROOT=/path/to/MathAgent
export PYTHON_BIN=python
export LD_LIBRARY_PATH=/path/to/python/lib:${LD_LIBRARY_PATH:-}
export PYTHONPATH=${MINT_SCOUT_ROOT}/src

"${PYTHON_BIN}" -m mint_scout.audit --config configs/tasks/casf2016_hpcc.yaml --output docs/AUDIT_REPORT_HPCC.md

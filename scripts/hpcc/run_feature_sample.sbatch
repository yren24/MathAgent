#!/bin/bash
#SBATCH --job-name=mint-scout-feature-sample
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --chdir=/path/to/MathAgent
#SBATCH --output=/path/to/MathAgent/logs/feature-sample-%j.out

set -euo pipefail

mkdir -p logs cache/plbind_features

export MINT_SCOUT_ROOT=/path/to/MathAgent
export PYTHON_BIN=python
export LD_LIBRARY_PATH=/path/to/python/lib:${LD_LIBRARY_PATH:-}
export PYTHONPATH=${MINT_SCOUT_ROOT}/src

INVARIANT=${INVARIANT:-PL}
SAMPLE_ID=${SAMPLE_ID:-10gs}

"${PYTHON_BIN}" -m mint_scout.compute_feature \
  --config configs/tasks/casf2016_hpcc.yaml \
  --invariant "${INVARIANT}" \
  --sample-id "${SAMPLE_ID}"

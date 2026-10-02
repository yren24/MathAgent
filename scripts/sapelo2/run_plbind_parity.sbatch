#!/bin/bash
#SBATCH --job-name=mint-agent-parity
#SBATCH --partition=batch
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16gb
#SBATCH --time=04:00:00
#SBATCH --output=/path/to/workdir/mathagent/logs/%x-%j.out
#SBATCH --error=/path/to/workdir/mathagent/logs/%x-%j.err

set -euo pipefail

export MINT_AGENT_ROOT=${MINT_AGENT_ROOT:-/path/to/MathAgent}
export PYTHON_BIN=${PYTHON_BIN:-python}
export PYTHONPATH=${MINT_AGENT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}

INVARIANT=${INVARIANT:-PL}
SAMPLE_ID=${SAMPLE_ID:-10gs}
RUN_ROOT=${RUN_ROOT:-/path/to/workdir/mathagent/runs}
CACHE_ROOT=${CACHE_ROOT:-/path/to/workdir/mathagent/cache/parity}

mkdir -p /path/to/workdir/mathagent/logs "${RUN_ROOT}" "${CACHE_ROOT}"
cd "${MINT_AGENT_ROOT}"
source scripts/sapelo2/load_modules.sh

"${PYTHON_BIN}" -m mint_scout.validate_plbind_parity \
  --sample-id "${SAMPLE_ID}" \
  --invariant "${INVARIANT}" \
  --legacy-root /path/to/legacy/embed_nn/plbind \
  --pdb-folder /path/to/data/all-pdbs \
  --output-root "${CACHE_ROOT}" \
  --report "${RUN_ROOT}/parity_${SAMPLE_ID}_${INVARIANT}_${SLURM_JOB_ID}.json"

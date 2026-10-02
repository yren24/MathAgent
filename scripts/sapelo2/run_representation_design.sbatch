#!/bin/bash
#SBATCH --job-name=mint-agent-repr
#SBATCH --partition=batch
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8gb
#SBATCH --time=02:00:00
#SBATCH --output=/path/to/workdir/mathagent/logs/%x-%j.out
#SBATCH --error=/path/to/workdir/mathagent/logs/%x-%j.err

set -euo pipefail

export MINT_AGENT_ROOT=${MINT_AGENT_ROOT:-/path/to/MathAgent}
export PYTHON_BIN=${PYTHON_BIN:-python}
export PYTHONPATH=${MINT_AGENT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}

SPLIT=${SPLIT:-train}
OFFSET=${OFFSET:-0}
LIMIT=${LIMIT:-64}
RUN_ROOT=${RUN_ROOT:-/path/to/workdir/mathagent/runs}

mkdir -p /path/to/workdir/mathagent/logs "${RUN_ROOT}"
cd "${MINT_AGENT_ROOT}"
source scripts/sapelo2/load_modules.sh

REPORT=${REPORT:-${RUN_ROOT}/representation_${SPLIT}_offset-${OFFSET}_limit-${LIMIT}_${SLURM_JOB_ID}.json}

if [[ "${LIMIT}" == "all" ]]; then
  "${PYTHON_BIN}" -m mint_scout.design_representation \
    --config configs/tasks/casf2016_sapelo2.yaml \
    --split "${SPLIT}" \
    --offset "${OFFSET}" \
    --output "${REPORT}"
else
  "${PYTHON_BIN}" -m mint_scout.design_representation \
    --config configs/tasks/casf2016_sapelo2.yaml \
    --split "${SPLIT}" \
    --offset "${OFFSET}" \
    --limit "${LIMIT}" \
    --output "${REPORT}"
fi

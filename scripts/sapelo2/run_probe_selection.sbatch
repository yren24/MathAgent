#!/bin/bash
#SBATCH --job-name=mint-agent-probe
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
export PYTHONPATH=${MINT_AGENT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}

REPRESENTATION_SPEC=${REPRESENTATION_SPEC:-/path/to/workdir/mathagent/runs/representation_train_offset-0_limit-all_47823391.json}
RUN_ROOT=${RUN_ROOT:-/path/to/workdir/mathagent/runs}

mkdir -p /path/to/workdir/mathagent/logs "${RUN_ROOT}"
cd "${MINT_AGENT_ROOT}"
source scripts/sapelo2/load_modules.sh

OUTPUT=${OUTPUT:-${RUN_ROOT}/probe_selection_${SLURM_JOB_ID}.json}

"${PYTHON_BIN}" -m mint_scout.select_casf_probe \
  --task-config configs/tasks/casf2016_sapelo2.yaml \
  --scout-config configs/scout/v1.yaml \
  --representation-spec "${REPRESENTATION_SPEC}" \
  --output "${OUTPUT}"

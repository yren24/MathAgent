#!/bin/bash
#SBATCH --job-name=mint-agent-probe-audit
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

RUN_ROOT=${RUN_ROOT:-/path/to/workdir/mathagent/runs}
PROBE_SELECTION=${PROBE_SELECTION:-/path/to/workdir/mathagent/runs/probe_selection_47979548.json}
OUTPUT=${OUTPUT:-${RUN_ROOT}/probe_audit_${SLURM_JOB_ID}.json}

mkdir -p /path/to/workdir/mathagent/logs "${RUN_ROOT}"
cd "${MINT_AGENT_ROOT}"
source scripts/sapelo2/load_modules.sh

"${PYTHON_BIN}" -m mint_scout.probe_audit \
  --task-config configs/tasks/casf2016_sapelo2.yaml \
  --scout-config configs/scout/v1.yaml \
  --probe-selection "${PROBE_SELECTION}" \
  --output "${OUTPUT}"

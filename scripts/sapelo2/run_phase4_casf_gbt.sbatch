#!/bin/bash
#SBATCH --job-name=mint-agent-full-gbt
#SBATCH --partition=batch
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64gb
#SBATCH --time=2-00:00:00
#SBATCH --output=/path/to/workdir/mathagent/logs/%x-%j.out
#SBATCH --error=/path/to/workdir/mathagent/logs/%x-%j.err

set -euo pipefail

export MINT_AGENT_ROOT=${MINT_AGENT_ROOT:-/path/to/MathAgent}
export PYTHON_BIN=${PYTHON_BIN:-python}
export PYTHONPATH=${MINT_AGENT_ROOT}/src

REPRESENTATION_SPEC=${REPRESENTATION_SPEC:-/path/to/workdir/mathagent/runs/representation_train_offset-0_limit-all_47823391.json}
SCOUT_ARTIFACT=${SCOUT_ARTIFACT:-/path/to/workdir/mathagent/runs/combined_scout_execution_47983663.json}
FEATURE_QC_REPORT=${FEATURE_QC_REPORT:-}
PL_MANIFEST=${PL_MANIFEST:-}
MAX_ACQUISITIONS=${MAX_ACQUISITIONS:-1}
RUN_ROOT=${RUN_ROOT:-/path/to/workdir/mathagent/runs}

if [[ -z "${FEATURE_QC_REPORT}" || -z "${PL_MANIFEST}" ]]; then
  echo "FEATURE_QC_REPORT and PL_MANIFEST are required" >&2
  exit 2
fi

mkdir -p /path/to/workdir/mathagent/logs "${RUN_ROOT}"
cd "${MINT_AGENT_ROOT}"
source scripts/sapelo2/load_modules.sh

"${PYTHON_BIN}" -m mint_scout.execute_casf \
  --config configs/tasks/casf2016_sapelo2.yaml \
  --gbt-config configs/gbt/plbind_fixed_gbt.yaml \
  --split train \
  --scout-artifact "${SCOUT_ARTIFACT}" \
  --representation-spec "${REPRESENTATION_SPEC}" \
  --feature-qc-report "${FEATURE_QC_REPORT}" \
  --feature-manifest "PL=${PL_MANIFEST}" \
  --max-acquisitions "${MAX_ACQUISITIONS}" \
  --output "${RUN_ROOT}/phase4_gbt_train_${SLURM_JOB_ID}.json"

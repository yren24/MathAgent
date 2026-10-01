#!/bin/bash
#SBATCH --job-name=mint-agent-outlier-diagnostic
#SBATCH --partition=batch
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8gb
#SBATCH --time=00:30:00
#SBATCH --output=/path/to/workdir/mathagent/logs/%x-%j.out
#SBATCH --error=/path/to/workdir/mathagent/logs/%x-%j.err

set -euo pipefail

export MINT_AGENT_ROOT=${MINT_AGENT_ROOT:-/path/to/MathAgent}
export PYTHON_BIN=${PYTHON_BIN:-python}
export PYTHONPATH=${MINT_AGENT_ROOT}/src

REPRESENTATION_SPEC=${REPRESENTATION_SPEC:-/path/to/workdir/mathagent/runs/representation_train_offset-0_limit-all_47823391.json}
PROBE_SELECTION=${PROBE_SELECTION:-/path/to/workdir/mathagent/runs/probe_selection_47979548.json}
INVARIANT=${INVARIANT:-FPRC}
FEATURE_MANIFEST=${FEATURE_MANIFEST:-/path/to/workdir/mathagent/runs/feature_batch_FPRC_train_offset-0_limit-all_47979669.jsonl}
QC_REPORT=${QC_REPORT:-/path/to/workdir/mathagent/runs/feature_qc_47983592.json}
RUN_ROOT=${RUN_ROOT:-/path/to/workdir/mathagent/runs}
OUTPUT=${OUTPUT:-${RUN_ROOT}/feature_outlier_diagnostic_${INVARIANT}_${SLURM_JOB_ID}.json}
TOP_K=${TOP_K:-10}

mkdir -p /path/to/workdir/mathagent/logs "${RUN_ROOT}"
cd "${MINT_AGENT_ROOT}"
source scripts/sapelo2/load_modules.sh

"${PYTHON_BIN}" -m mint_scout.feature_outlier_diagnostic \
  --representation-spec "${REPRESENTATION_SPEC}" \
  --probe-selection "${PROBE_SELECTION}" \
  --feature-manifest "${INVARIANT}=${FEATURE_MANIFEST}" \
  --qc-report "${QC_REPORT}" \
  --top-k "${TOP_K}" \
  --dataset-id casf2016 \
  --evidence-scope probe \
  --output "${OUTPUT}"

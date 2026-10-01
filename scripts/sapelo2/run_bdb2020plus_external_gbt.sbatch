#!/bin/bash
#SBATCH --job-name=mint-agent-bdb-gbt
#SBATCH --partition=batch
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64gb
#SBATCH --time=2-00:00:00
#SBATCH --output=/path/to/workdir/mathagent/logs/%x-%j.out
#SBATCH --error=/path/to/workdir/mathagent/logs/%x-%j.err

set -euo pipefail

MINT_AGENT_ROOT=${MINT_AGENT_ROOT:-/path/to/MathAgent}
EXTERNAL_ROOT=${EXTERNAL_ROOT:-/path/to/workdir/mathagent/external/bdb2020plus}
PYTHON_BIN=${PYTHON_BIN:-python}
REPRESENTATION_SPEC=${REPRESENTATION_SPEC:-/path/to/workdir/mathagent/runs/representation_train_offset-0_limit-all_47823391.json}
TRAIN_PL_MANIFEST=${TRAIN_PL_MANIFEST:-/path/to/workdir/mathagent/runs/feature_batch_PL_train_offset-0_limit-all_47988267.jsonl}
TRAIN_QC_REPORT=${TRAIN_QC_REPORT:-/path/to/workdir/mathagent/runs/feature_qc_47988575.json}
EXTERNAL_PL_MANIFEST=${EXTERNAL_PL_MANIFEST:?EXTERNAL_PL_MANIFEST must be provided}
EXTERNAL_QC_REPORT=${EXTERNAL_QC_REPORT:?EXTERNAL_QC_REPORT must be provided}
N_ESTIMATORS=${N_ESTIMATORS:-4000}
N_RUNS=${N_RUNS:-10}
RUN_KIND=${RUN_KIND:-formal_evaluation}

mkdir -p "${EXTERNAL_ROOT}/runs" /path/to/workdir/mathagent/logs
cd "${MINT_AGENT_ROOT}"
source scripts/sapelo2/load_modules.sh
export PYTHONPATH=${MINT_AGENT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}

"${PYTHON_BIN}" -m mint_scout.evaluate_external_gbt \
  --training-task-config configs/tasks/casf2016_sapelo2.yaml \
  --external-task-config configs/tasks/bdb2020plus_sapelo2.yaml \
  --representation-spec "${REPRESENTATION_SPEC}" \
  --train-feature-manifest "PL=${TRAIN_PL_MANIFEST}" \
  --external-feature-manifest "PL=${EXTERNAL_PL_MANIFEST}" \
  --train-feature-qc-report "${TRAIN_QC_REPORT}" \
  --external-feature-qc-report "${EXTERNAL_QC_REPORT}" \
  --n-estimators "${N_ESTIMATORS}" \
  --n-runs "${N_RUNS}" \
  --run-kind "${RUN_KIND}" \
  --n-bootstrap 500 \
  --bootstrap-seed 2026 \
  --output "${EXTERNAL_ROOT}/runs/external_gbt_PL_${SLURM_JOB_ID}.json"

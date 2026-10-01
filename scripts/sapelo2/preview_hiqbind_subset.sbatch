#!/bin/bash
#SBATCH --job-name=mint-hiqbind-preview
#SBATCH --partition=batch
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4gb
#SBATCH --time=00:15:00
#SBATCH --output=/path/to/workdir/mathagent/logs/%x-%j.out
#SBATCH --error=/path/to/workdir/mathagent/logs/%x-%j.err

set -euo pipefail

MINT_AGENT_ROOT=${MINT_AGENT_ROOT:-/path/to/MathAgent}
PYTHON_BIN=${PYTHON_BIN:-${MINT_AGENT_ROOT}/.venv-langgraph/bin/python}
METADATA=${METADATA:-/path/to/workdir/mathagent/external/hiqbind/source/hiqbind_sm_metadata.csv}
OUTPUT=${OUTPUT:-/path/to/workdir/mathagent/runs/hiqbind_subset_preview.json}
MAX_SAMPLES=${MAX_SAMPLES:-500}
SELECTION_STRATEGY=${SELECTION_STRATEGY:-stratified}
RANDOM_SEED=${RANDOM_SEED:-2026}
TARGET_QUANTILE_BINS=${TARGET_QUANTILE_BINS:-10}
LIGAND_SIZE_QUANTILE_BINS=${LIGAND_SIZE_QUANTILE_BINS:-5}
MAX_PER_PDB_ID=${MAX_PER_PDB_ID:-1}

cd "${MINT_AGENT_ROOT}"
source scripts/sapelo2/load_modules.sh
unset EBPYTHONPREFIXES
export PYTHONPATH=${MINT_AGENT_ROOT}/src

"${PYTHON_BIN}" -m mint_scout.data.hiqbind preview-subset \
  --metadata "${METADATA}" \
  --measurement kd \
  --measurement ki \
  --affinity-sign = \
  --invalid-affinity-policy skip \
  --max-samples "${MAX_SAMPLES}" \
  --strategy "${SELECTION_STRATEGY}" \
  --random-seed "${RANDOM_SEED}" \
  --target-quantile-bins "${TARGET_QUANTILE_BINS}" \
  --ligand-size-quantile-bins "${LIGAND_SIZE_QUANTILE_BINS}" \
  --max-per-pdb-id "${MAX_PER_PDB_ID}" \
  --output "${OUTPUT}"

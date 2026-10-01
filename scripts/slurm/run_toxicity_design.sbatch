#!/bin/bash
#SBATCH --job-name=mint-tox-design
#SBATCH --partition=batch
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/path/to/workdir/mathagent/logs/toxicity-design-%j.out
#SBATCH --error=/path/to/workdir/mathagent/logs/toxicity-design-%j.err

set -euo pipefail

MINT_AGENT_ROOT=${MINT_AGENT_ROOT:-/path/to/MathAgent}
DATASET_DIR=${DATASET_DIR:-/path/to/workdir/mathagent/external/toxicity-phase0/LD50}
DATASET_ID=${DATASET_ID:-LD50}
MANIFEST_PATH=${MANIFEST_PATH:-}
OUTPUT_ROOT=${OUTPUT_ROOT:-/path/to/workdir/mathagent/toxicity/design-v1}
AGENT_ENV=${AGENT_ENV:-/path/to/MathAgent/.venv-langgraph}
PYTHON_BIN=${PYTHON_BIN:-${AGENT_ENV}/bin/python}
GUDHI_ROOT=${GUDHI_ROOT:-/path/to/workdir/mathagent/deps/gudhi-3.13.0}

mkdir -p /path/to/workdir/mathagent/logs "${OUTPUT_ROOT}"
cd "${MINT_AGENT_ROOT}"
source scripts/sapelo2/load_modules.sh
unset EBPYTHONPREFIXES
export PYTHONPATH="${MINT_AGENT_ROOT}/src:${GUDHI_ROOT}"
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1

"${PYTHON_BIN}" -m pytest -q \
  tests/test_toxicity_legacy.py \
  tests/test_toxicity_design.py

if [[ -n "${MANIFEST_PATH}" ]]; then
  cp "${MANIFEST_PATH}" "${OUTPUT_ROOT}/ld50_manifest.csv"
else
  "${PYTHON_BIN}" -m mint_scout.toxicity.manifest \
    --dataset-dir "${DATASET_DIR}" \
    --output "${OUTPUT_ROOT}/ld50_manifest.csv"
fi

"${PYTHON_BIN}" -m mint_scout.toxicity.design \
  --manifest "${OUTPUT_ROOT}/ld50_manifest.csv" \
  --dataset-id "${DATASET_ID}" \
  --primary-metric "${PRIMARY_METRIC:-PCC2}" \
  --output-dir "${OUTPUT_ROOT}"

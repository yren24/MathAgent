#!/bin/bash
#SBATCH --job-name=mint-tox-parity
#SBATCH --partition=batch
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8gb
#SBATCH --time=01:00:00
#SBATCH --output=/path/to/workdir/mathagent/logs/%x-%j.out
#SBATCH --error=/path/to/workdir/mathagent/logs/%x-%j.err

set -euo pipefail

MINT_AGENT_ROOT=${MINT_AGENT_ROOT:-/path/to/MathAgent}
TOXICITY_BUNDLE=${TOXICITY_BUNDLE:-/path/to/workdir/mathagent/external/toxicity-phase0}
OUTPUT_ROOT=${OUTPUT_ROOT:-/path/to/workdir/mathagent/toxicity/phase1-parity}
AGENT_ENV=${AGENT_ENV:-/path/to/MathAgent/.venv-langgraph}
PYTHON_BIN=${PYTHON_BIN:-${AGENT_ENV}/bin/python}
GUDHI_ROOT=${GUDHI_ROOT:-/path/to/workdir/mathagent/deps/gudhi-3.13.0}

mkdir -p /path/to/workdir/mathagent/logs "${OUTPUT_ROOT}"
cd "${MINT_AGENT_ROOT}"
source scripts/sapelo2/load_modules.sh
unset EBPYTHONPREFIXES
export PYTHONPATH=${MINT_AGENT_ROOT}/src:${GUDHI_ROOT}
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1

"${PYTHON_BIN}" -m pytest -q tests/test_toxicity_legacy.py

for sample_spec in "train:100-01-6" "test:100-00-5"; do
  split=${sample_spec%%:*}
  sample_id=${sample_spec#*:}
  "${PYTHON_BIN}" -m mint_scout.toxicity.parity \
    --sample-id "${sample_id}" \
    --split "${split}" \
    --invariant PH \
    --invariant PL \
    --invariant CA \
    --invariant FPRC \
    --invariant EIC \
    --legacy-root "${TOXICITY_BUNDLE}/legacy" \
    --dataset-dir "${TOXICITY_BUNDLE}/LD50" \
    --reference-root "${TOXICITY_BUNDLE}/LD50/topology_features/maxfil10" \
    --output-root "${OUTPUT_ROOT}/generated" \
    --report "${OUTPUT_ROOT}/${split}-${sample_id}-parity.json"
done

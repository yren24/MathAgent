#!/bin/bash
#SBATCH --job-name=mint-tox-probe
#SBATCH --partition=batch
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/path/to/workdir/mathagent/logs/toxicity-probe-%j.out
#SBATCH --error=/path/to/workdir/mathagent/logs/toxicity-probe-%j.err

set -euo pipefail

MINT_AGENT_ROOT=${MINT_AGENT_ROOT:-/path/to/MathAgent}
MANIFEST_ROOT=${MANIFEST_ROOT:-/path/to/workdir/mathagent/toxicity/design-v1}
DESIGN_ROOT=${DESIGN_ROOT:-/path/to/workdir/mathagent/toxicity/design-v1-advisory}
OUTPUT_ROOT=${OUTPUT_ROOT:-/path/to/workdir/mathagent/toxicity/probe-v1}
ADVISORY_ARTIFACT=${ADVISORY_ARTIFACT:-}
DATASET_ID=${DATASET_ID:-LD50}
AGENT_ENV=${AGENT_ENV:-/path/to/MathAgent/.venv-langgraph}
PYTHON_BIN=${PYTHON_BIN:-${AGENT_ENV}/bin/python}
GUDHI_ROOT=${GUDHI_ROOT:-/path/to/workdir/mathagent/deps/gudhi-3.13.0}

mkdir -p /path/to/workdir/mathagent/logs "${OUTPUT_ROOT}"
cd "${MINT_AGENT_ROOT}"
source scripts/sapelo2/load_modules.sh
unset EBPYTHONPREFIXES
export PYTHONPATH="${MINT_AGENT_ROOT}/src:${GUDHI_ROOT}"
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1

"${PYTHON_BIN}" -m pytest -q tests/test_toxicity_probe.py

args=(
  --manifest "${MANIFEST_ROOT}/ld50_manifest.csv" \
  --dataset-id "${DATASET_ID}" \
  --design-report "${DESIGN_ROOT}/toxicity_representation_design.json" \
  --representation-dir "${DESIGN_ROOT}/representations" \
  --output "${OUTPUT_ROOT}/toxicity_probe_selection.json"
)
if [[ -n "${ADVISORY_ARTIFACT}" ]]; then
  args+=(--advisory-artifact "${ADVISORY_ARTIFACT}")
fi

"${PYTHON_BIN}" -m mint_scout.toxicity.probe "${args[@]}"

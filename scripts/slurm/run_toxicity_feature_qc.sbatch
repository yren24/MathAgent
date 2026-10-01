#!/bin/bash
#SBATCH --job-name=mint-tox-qc
#SBATCH --partition=batch
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=01:00:00
#SBATCH --output=/path/to/workdir/mathagent/logs/toxicity-feature-qc-%j.out
#SBATCH --error=/path/to/workdir/mathagent/logs/toxicity-feature-qc-%j.err

set -euo pipefail

MINT_AGENT_ROOT=${MINT_AGENT_ROOT:-/path/to/MathAgent}
FEATURE_ROOT=${FEATURE_ROOT:-/path/to/workdir/mathagent/toxicity/probe-v1/features}
PLAN_PATH=${PLAN_PATH:-${FEATURE_ROOT}/toxicity_feature_plan.json}
QC_OUTPUT=${QC_OUTPUT:-${FEATURE_ROOT}/toxicity_feature_qc.json}
EXPECTED_SAMPLE_COUNT=${EXPECTED_SAMPLE_COUNT:-}
AGENT_ENV=${AGENT_ENV:-/path/to/MathAgent/.venv-langgraph}
PYTHON_BIN=${PYTHON_BIN:-${AGENT_ENV}/bin/python}

mkdir -p /path/to/workdir/mathagent/logs "${FEATURE_ROOT}"
cd "${MINT_AGENT_ROOT}"
source scripts/sapelo2/load_modules.sh
unset EBPYTHONPREFIXES
export PYTHONPATH="${MINT_AGENT_ROOT}/src"
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1

"${PYTHON_BIN}" -m pytest -q tests/test_toxicity_feature_qc.py

qc_args=(
  --plan "${PLAN_PATH}"
  --output "${QC_OUTPUT}"
)
if [[ -n "${EXPECTED_SAMPLE_COUNT}" ]]; then
  qc_args+=(--expected-sample-count "${EXPECTED_SAMPLE_COUNT}")
fi

"${PYTHON_BIN}" -m mint_scout.toxicity.feature_qc \
  "${qc_args[@]}"

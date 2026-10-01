#!/bin/bash
#SBATCH --job-name=mint-tox-feat-plan
#SBATCH --partition=batch
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#SBATCH --output=/path/to/workdir/mathagent/logs/toxicity-feature-plan-%j.out
#SBATCH --error=/path/to/workdir/mathagent/logs/toxicity-feature-plan-%j.err

set -euo pipefail

MINT_AGENT_ROOT=${MINT_AGENT_ROOT:-/path/to/MathAgent}
DESIGN_ROOT=${DESIGN_ROOT:-/path/to/workdir/mathagent/toxicity/design-v1-advisory}
OUTPUT_ROOT=${OUTPUT_ROOT:-/path/to/workdir/mathagent/toxicity/probe-v1/features}
AGENT_ENV=${AGENT_ENV:-/path/to/MathAgent/.venv-langgraph}
PYTHON_BIN=${PYTHON_BIN:-${AGENT_ENV}/bin/python}

mkdir -p /path/to/workdir/mathagent/logs "${OUTPUT_ROOT}/manifests"
cd "${MINT_AGENT_ROOT}"
source scripts/sapelo2/load_modules.sh
unset EBPYTHONPREFIXES
export PYTHONPATH="${MINT_AGENT_ROOT}/src"
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1

"${PYTHON_BIN}" -m pytest -q \
  tests/test_toxicity_feature_plan.py \
  tests/test_toxicity_probe.py

task_config_args=()
if [[ -n "${TASK_CONFIG:-}" ]]; then
  task_config_args=(--task-config "${TASK_CONFIG}")
fi

"${PYTHON_BIN}" -m mint_scout.toxicity.feature_plan \
  --design-report "${DESIGN_ROOT}/toxicity_representation_design.json" \
  --representation-dir "${DESIGN_ROOT}/representations" \
  --manifest-dir "${OUTPUT_ROOT}/manifests" \
  "${task_config_args[@]}" \
  --output "${OUTPUT_ROOT}/toxicity_feature_plan.json"

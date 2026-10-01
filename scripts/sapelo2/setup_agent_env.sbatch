#!/bin/bash
#SBATCH --job-name=mint-agent-env
#SBATCH --partition=batch
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4gb
#SBATCH --time=00:30:00
#SBATCH --output=/path/to/workdir/mathagent/logs/%x-%j.out
#SBATCH --error=/path/to/workdir/mathagent/logs/%x-%j.err

set -euo pipefail

export MINT_AGENT_ROOT=${MINT_AGENT_ROOT:-/path/to/MathAgent}
AGENT_ENV=${AGENT_ENV:-${MINT_AGENT_ROOT}/.venv-langgraph}

mkdir -p /path/to/workdir/mathagent/logs
cd "${MINT_AGENT_ROOT}"
source scripts/sapelo2/load_modules.sh

if [[ ! -x "${AGENT_ENV}/bin/python" ]]; then
  python -m venv "${AGENT_ENV}"
fi

unset EBPYTHONPREFIXES
unset PYTHONPATH

"${AGENT_ENV}/bin/python" -m pip install --upgrade pip setuptools wheel
"${AGENT_ENV}/bin/python" -m pip install -e "${MINT_AGENT_ROOT}[dev,agent,datasets]"
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=${MINT_AGENT_ROOT}/src \
  "${AGENT_ENV}/bin/python" -m pytest -q \
  tests/test_agent_graph.py \
  tests/test_artifact_memory.py \
  tests/test_representation_comparison.py

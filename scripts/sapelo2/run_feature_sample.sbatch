#!/bin/bash
#SBATCH --job-name=mint-agent-feature-sample
#SBATCH --partition=batch
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16gb
#SBATCH --time=04:00:00
#SBATCH --output=/path/to/workdir/mathagent/logs/%x-%j.out
#SBATCH --error=/path/to/workdir/mathagent/logs/%x-%j.err

set -euo pipefail

export MINT_AGENT_ROOT=${MINT_AGENT_ROOT:-/path/to/MathAgent}
export PYTHON_BIN=${PYTHON_BIN:-python}
export PYTHONPATH=${MINT_AGENT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}

INVARIANT=${INVARIANT:-PL}
SAMPLE_ID=${SAMPLE_ID:-10gs}
REPRESENTATION_SPEC=${REPRESENTATION_SPEC:-}

mkdir -p /path/to/workdir/mathagent/logs /path/to/workdir/mathagent/cache/plbind_features
cd "${MINT_AGENT_ROOT}"
source scripts/sapelo2/load_modules.sh

args=(
  --config configs/tasks/casf2016_sapelo2.yaml
  --invariant "${INVARIANT}"
  --sample-id "${SAMPLE_ID}"
)
if [[ -n "${REPRESENTATION_SPEC}" ]]; then
  args+=(--representation-spec "${REPRESENTATION_SPEC}")
fi

"${PYTHON_BIN}" -m mint_scout.compute_feature "${args[@]}"

#!/bin/bash
#SBATCH --job-name=mint-agent-register
#SBATCH --partition=batch
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1gb
#SBATCH --time=00:10:00
#SBATCH --output=/path/to/workdir/mathagent/logs/%x-%j.out
#SBATCH --error=/path/to/workdir/mathagent/logs/%x-%j.err

set -euo pipefail

export MINT_AGENT_ROOT=${MINT_AGENT_ROOT:-/path/to/MathAgent}
export PYTHON_BIN=${PYTHON_BIN:-python}
export PYTHONPATH=${MINT_AGENT_ROOT}/src

REGISTRY=${REGISTRY:-/path/to/workdir/mathagent/memory/artifacts.sqlite}

cd "${MINT_AGENT_ROOT}"
source scripts/sapelo2/load_modules.sh

if [[ -n "${BATCH_SPEC:-}" ]]; then
  "${PYTHON_BIN}" -m mint_scout.artifact_memory register-batch --spec "${BATCH_SPEC}"
  exit 0
fi

: "${ARTIFACT:?Set ARTIFACT for one file or BATCH_SPEC for a batch registration spec}"

args=(
  register
  --registry "${REGISTRY}"
  --artifact "${ARTIFACT}"
)
[[ -n "${KIND:-}" ]] && args+=(--kind "${KIND}")
[[ -n "${DATASET_ID:-}" ]] && args+=(--dataset-id "${DATASET_ID}")
[[ -n "${SPLIT:-}" ]] && args+=(--split "${SPLIT}")
[[ -n "${EVIDENCE_SCOPE:-}" ]] && args+=(--evidence-scope "${EVIDENCE_SCOPE}")
[[ -n "${INVARIANT:-}" ]] && args+=(--invariant "${INVARIANT}")
[[ -n "${REPRESENTATION_HASH:-}" ]] && args+=(--representation-hash "${REPRESENTATION_HASH}")
[[ -n "${SELECTION_HASH:-}" ]] && args+=(--selection-hash "${SELECTION_HASH}")
[[ -n "${STATUS:-}" ]] && args+=(--status "${STATUS}")

"${PYTHON_BIN}" -m mint_scout.artifact_memory "${args[@]}"

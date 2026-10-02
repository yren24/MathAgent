#!/bin/bash
#SBATCH --job-name=mint-register-artifact
#SBATCH --partition=batch
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1gb
#SBATCH --time=00:05:00
#SBATCH --output=/path/to/workdir/mathagent/logs/%x-%j.out
#SBATCH --error=/path/to/workdir/mathagent/logs/%x-%j.err

set -euo pipefail

MINT_AGENT_ROOT=${MINT_AGENT_ROOT:-/path/to/MathAgent}
PYTHON_BIN=${PYTHON_BIN:-python}
REGISTRY=${REGISTRY:-/path/to/workdir/mathagent/memory/artifacts.sqlite}
: "${ARTIFACT:?ARTIFACT is required}"

cd "${MINT_AGENT_ROOT}"
source scripts/sapelo2/load_modules.sh
unset EBPYTHONPREFIXES
export PYTHONPATH=${MINT_AGENT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}

args=(register --registry "${REGISTRY}" --artifact "${ARTIFACT}")
[[ -n "${ARTIFACT_KIND:-}" ]] && args+=(--kind "${ARTIFACT_KIND}")
[[ -n "${DATASET_ID:-}" ]] && args+=(--dataset-id "${DATASET_ID}")
[[ -n "${SPLIT:-}" ]] && args+=(--split "${SPLIT}")
[[ -n "${EVIDENCE_SCOPE:-}" ]] && args+=(--evidence-scope "${EVIDENCE_SCOPE}")
[[ -n "${REPRESENTATION_HASH:-}" ]] && args+=(--representation-hash "${REPRESENTATION_HASH}")
[[ -n "${SELECTION_HASH:-}" ]] && args+=(--selection-hash "${SELECTION_HASH}")
[[ -n "${REPORT_STATUS:-}" ]] && args+=(--status "${REPORT_STATUS}")

"${PYTHON_BIN}" -m mint_scout.artifact_memory "${args[@]}"

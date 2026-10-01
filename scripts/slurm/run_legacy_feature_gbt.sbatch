#!/bin/bash
#SBATCH --job-name=mint-legacy-feature-gbt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

set -euo pipefail

MINT_AGENT_ROOT=${MINT_AGENT_ROOT:?MINT_AGENT_ROOT must be provided}
INDEX_ROOT=${INDEX_ROOT:?INDEX_ROOT must be provided}
FEATURE_DIR=${FEATURE_DIR:?FEATURE_DIR must be provided}
INVARIANT=${INVARIANT:?INVARIANT must be provided}
GBT_CONFIG=${GBT_CONFIG:?GBT_CONFIG must be provided}
OUTPUT=${OUTPUT:?OUTPUT must be provided}
MINT_MODULE_SETUP=${MINT_MODULE_SETUP:?MINT_MODULE_SETUP must be provided}

source "${MINT_MODULE_SETUP}"
cd "${MINT_AGENT_ROOT}"
export PYTHONPATH=${MINT_AGENT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}
python scripts/diagnostics/evaluate_legacy_feature_gbt.py \
  --index-root "${INDEX_ROOT}" \
  --feature-dir "${FEATURE_DIR}" \
  --invariant "${INVARIANT}" \
  --gbt-config "${GBT_CONFIG}" \
  --output "${OUTPUT}"

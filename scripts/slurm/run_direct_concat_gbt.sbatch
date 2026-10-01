#!/bin/bash
#SBATCH --job-name=mint-direct-concat-gbt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

set -euo pipefail

MINT_AGENT_ROOT=${MINT_AGENT_ROOT:?MINT_AGENT_ROOT must be provided}
TASK_CONFIG=${TASK_CONFIG:?TASK_CONFIG must be provided}
GBT_CONFIG=${GBT_CONFIG:?GBT_CONFIG must be provided}
REPRESENTATION_SPEC=${REPRESENTATION_SPEC:?REPRESENTATION_SPEC must be provided}
FEATURE_INVARIANTS=${FEATURE_INVARIANTS:?FEATURE_INVARIANTS must be provided}
OUTPUT=${OUTPUT:?OUTPUT must be provided}

cd "${MINT_AGENT_ROOT}"
export PYTHONPATH=${MINT_AGENT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}
if [[ -n "${MINT_MODULE_SETUP:-}" ]]; then
  source "${MINT_MODULE_SETUP}"
fi

args=(
  --task-config "${TASK_CONFIG}"
  --representation-spec "${REPRESENTATION_SPEC}"
  --gbt-config "${GBT_CONFIG}"
  --invariants "${FEATURE_INVARIANTS}"
  --output "${OUTPUT}"
)
IFS=':' read -r -a invariants <<< "${FEATURE_INVARIANTS}"
for invariant in "${invariants[@]}"; do
  train_var="FEATURE_MANIFEST_TRAIN_${invariant}"
  test_var="FEATURE_MANIFEST_TEST_${invariant}"
  args+=(--train-feature-manifest "${invariant}=${!train_var}")
  args+=(--test-feature-manifest "${invariant}=${!test_var}")
done

python -m mint_scout.evaluate_direct_concat_gbt "${args[@]}"

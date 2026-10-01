#!/bin/bash
#SBATCH --job-name=mint-agent-acceptance-evaluation
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

set -euo pipefail

MINT_AGENT_ROOT=${MINT_AGENT_ROOT:?MINT_AGENT_ROOT must be provided}
RUN_ROOT=${RUN_ROOT:-${MINT_AGENT_ROOT}/runs}
PYTHON_BIN=${PYTHON_BIN:-python}
TASK_CONFIG=${TASK_CONFIG:?TASK_CONFIG must be provided}
GBT_CONFIG=${GBT_CONFIG:?GBT_CONFIG must be provided}
REPRESENTATION_SPEC=${REPRESENTATION_SPEC:?REPRESENTATION_SPEC must be provided}
SCOUT_ARTIFACT=${SCOUT_ARTIFACT:?SCOUT_ARTIFACT must be provided}
ACCEPTANCE_MODE=${ACCEPTANCE_MODE:-candidate}
TRAIN_QC_REPORT=${TRAIN_QC_REPORT:?TRAIN_QC_REPORT must be provided}
EVALUATION_QC_REPORT=${EVALUATION_QC_REPORT:?EVALUATION_QC_REPORT must be provided}
FEATURE_INVARIANTS=${FEATURE_INVARIANTS:?FEATURE_INVARIANTS must be provided}
OUTPUT=${OUTPUT:?OUTPUT must be provided}

cd "${MINT_AGENT_ROOT}"
export PYTHONPATH=${MINT_AGENT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}

if [[ -n "${MINT_MODULE_SETUP:-}" ]]; then
  source "${MINT_MODULE_SETUP}"
elif [[ -n "${MINT_MODULES:-}" ]]; then
  module purge
  IFS=':' read -r -a mint_modules <<< "${MINT_MODULES}"
  for module_name in "${mint_modules[@]}"; do
    module load "${module_name}"
  done
fi

args=(
  --task-config "${TASK_CONFIG}"
  --gbt-config "${GBT_CONFIG}"
  --representation-spec "${REPRESENTATION_SPEC}"
  --scout-artifact "${SCOUT_ARTIFACT}"
  --train-feature-qc-report "${TRAIN_QC_REPORT}"
  --evaluation-feature-qc-report "${EVALUATION_QC_REPORT}"
  --output "${OUTPUT}"
)
if [[ "${ACCEPTANCE_MODE}" == "progressive" ]]; then
  args+=(--max-acquisitions "${MAX_ACQUISITIONS:?MAX_ACQUISITIONS must be provided}")
  if [[ -n "${PRIOR_EVALUATION_REPORT:-}" ]]; then
    args+=(--prior-evaluation-report "${PRIOR_EVALUATION_REPORT}")
  fi
else
  args+=(--candidate-rank "${CANDIDATE_RANK:?CANDIDATE_RANK must be provided}")
fi
IFS=':' read -r -a feature_invariants <<< "${FEATURE_INVARIANTS}"
for invariant in "${feature_invariants[@]}"; do
  train_name="FEATURE_MANIFEST_TRAIN_${invariant}"
  evaluation_name="FEATURE_MANIFEST_EVALUATION_${invariant}"
  args+=(--train-feature-manifest "${invariant}=${!train_name}")
  args+=(--evaluation-feature-manifest "${invariant}=${!evaluation_name}")
done

"${PYTHON_BIN}" -m mint_scout.evaluate_acceptance_gbt "${args[@]}"

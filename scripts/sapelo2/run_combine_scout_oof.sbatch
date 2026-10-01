#!/bin/bash
#SBATCH --job-name=mint-agent-combine
#SBATCH --partition=batch
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8gb
#SBATCH --time=01:00:00
#SBATCH --output=/path/to/workdir/mathagent/logs/%x-%j.out
#SBATCH --error=/path/to/workdir/mathagent/logs/%x-%j.err

set -euo pipefail

export MINT_AGENT_ROOT=${MINT_AGENT_ROOT:-/path/to/MathAgent}
export PYTHON_BIN=${PYTHON_BIN:-python}
export PYTHONPATH=${MINT_AGENT_ROOT}/src

REPRESENTATION_SPEC=${REPRESENTATION_SPEC:-/path/to/workdir/mathagent/runs/representation_train_offset-0_limit-all_47823391.json}
PROBE_SELECTION=${PROBE_SELECTION:-/path/to/workdir/mathagent/runs/probe_selection_47979548.json}
RUN_ROOT=${RUN_ROOT:-/path/to/workdir/mathagent/runs}
USER_TARGET=${USER_TARGET:-}
FEATURE_QC_REPORT=${FEATURE_QC_REPORT:-}

mkdir -p /path/to/workdir/mathagent/logs "${RUN_ROOT}"
cd "${MINT_AGENT_ROOT}"
source scripts/sapelo2/load_modules.sh

args=(
  --scout-config configs/scout/v1.yaml
  --representation-spec "${REPRESENTATION_SPEC}"
  --probe-selection "${PROBE_SELECTION}"
  --output "${RUN_ROOT}/combined_scout_${SLURM_JOB_ID}.json"
  --execution-artifact "${RUN_ROOT}/combined_scout_execution_${SLURM_JOB_ID}.json"
)
for invariant in PH PL CA FPRC EIC; do
  variable_name="${invariant}_SCOUT_REPORT"
  report=${!variable_name:-}
  if [[ -n "${report}" ]]; then
    args+=(--scout-report "${report}")
  fi
done
if [[ -n "${USER_TARGET}" ]]; then
  args+=(--user-target "${USER_TARGET}")
fi
if [[ -n "${FEATURE_QC_REPORT}" ]]; then
  args+=(--feature-qc-report "${FEATURE_QC_REPORT}")
fi

"${PYTHON_BIN}" -m mint_scout.combine_scout_oof "${args[@]}"

#!/bin/bash
#SBATCH --job-name=mint-agent-filtration-audit
#SBATCH --partition=batch
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16gb
#SBATCH --time=02:00:00
#SBATCH --output=/path/to/workdir/mathagent/logs/%x-%j.out
#SBATCH --error=/path/to/workdir/mathagent/logs/%x-%j.err

set -euo pipefail

export MINT_AGENT_ROOT=${MINT_AGENT_ROOT:-/path/to/MathAgent}
export PYTHON_BIN=${PYTHON_BIN:-python}
export PYTHONPATH=${MINT_AGENT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}

SCOUT_CONFIG=${SCOUT_CONFIG:-configs/scout/v1.yaml}
REPRESENTATION_SPEC=${REPRESENTATION_SPEC:-/path/to/workdir/mathagent/runs/representation_train_offset-0_limit-all_47823391.json}
PROBE_SELECTION=${PROBE_SELECTION:-/path/to/workdir/mathagent/runs/probe_selection_47979548.json}
RUN_ROOT=${RUN_ROOT:-/path/to/workdir/mathagent/runs}
OUTPUT=${OUTPUT:-${RUN_ROOT}/filtration_audit_${SLURM_JOB_ID}.json}

mkdir -p /path/to/workdir/mathagent/logs "${RUN_ROOT}"
cd "${MINT_AGENT_ROOT}"
source scripts/sapelo2/load_modules.sh

args=(
  --representation-spec "${REPRESENTATION_SPEC}"
  --probe-selection "${PROBE_SELECTION}"
  --audit-config "${SCOUT_CONFIG}"
  --output "${OUTPUT}"
)
for invariant in PH PL CA FPRC EIC; do
  variable_name="${invariant}_MANIFEST"
  manifest=${!variable_name:-}
  if [[ -n "${manifest}" ]]; then
    args+=(--feature-manifest "${invariant}=${manifest}")
  fi
done

"${PYTHON_BIN}" -m mint_scout.filtration_audit "${args[@]}"

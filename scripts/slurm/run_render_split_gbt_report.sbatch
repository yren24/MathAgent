#!/bin/bash
#SBATCH --job-name=mint-agent-split-report
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2gb
#SBATCH --time=00:10:00

set -euo pipefail

MINT_AGENT_ROOT=${MINT_AGENT_ROOT:?MINT_AGENT_ROOT must be provided}
INPUT=${INPUT:?INPUT must be provided}
OUTPUT=${OUTPUT:?OUTPUT must be provided}
PYTHON_BIN=${PYTHON_BIN:-python}

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

"${PYTHON_BIN}" -m mint_scout.render_split_gbt_report \
  --input "${INPUT}" \
  --output "${OUTPUT}"

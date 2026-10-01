#!/bin/bash
#SBATCH --job-name=mint-probe-ablation-summary
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

set -euo pipefail

MINT_AGENT_ROOT=${MINT_AGENT_ROOT:?MINT_AGENT_ROOT must be provided by the execution profile}
PYTHON_BIN=${PYTHON_BIN:-python}
ARM_SPECS=${ARM_SPECS:?ARM_SPECS must provide NAME=SELECTION,AUDIT,SCOUT entries separated by semicolons}
BASELINE_ARM=${BASELINE_ARM:?BASELINE_ARM must name one arm}
SUMMARY_OUTPUT=${SUMMARY_OUTPUT:?SUMMARY_OUTPUT must be provided}
MARKDOWN_OUTPUT=${MARKDOWN_OUTPUT:-}

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

mkdir -p "$(dirname "${SUMMARY_OUTPUT}")"
args=(--baseline "${BASELINE_ARM}" --output "${SUMMARY_OUTPUT}")
if [[ -n "${MARKDOWN_OUTPUT}" ]]; then
  args+=(--markdown-output "${MARKDOWN_OUTPUT}")
fi

IFS=';' read -r -a arms <<< "${ARM_SPECS}"
for arm in "${arms[@]}"; do
  args+=(--arm "${arm}")
done

"${PYTHON_BIN}" -m mint_scout.summarize_probe_ablation "${args[@]}"

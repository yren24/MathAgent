#!/bin/bash
#SBATCH --job-name=mint-filtration-compare
#SBATCH --partition=batch
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2gb
#SBATCH --time=00:10:00
#SBATCH --output=/path/to/workdir/mathagent/logs/%x-%j.out
#SBATCH --error=/path/to/workdir/mathagent/logs/%x-%j.err

set -euo pipefail

MINT_AGENT_ROOT=${MINT_AGENT_ROOT:-/path/to/MathAgent}
PYTHON_BIN=${PYTHON_BIN:-python}
: "${BASELINE_REPRESENTATION:?BASELINE_REPRESENTATION is required}"
: "${CANDIDATE_REPRESENTATION:?CANDIDATE_REPRESENTATION is required}"
: "${BASELINE_OOF:?BASELINE_OOF is required as a comma-separated list}"
: "${CANDIDATE_OOF:?CANDIDATE_OOF is required as a comma-separated list}"
: "${MIN_PCC_DELTA:?MIN_PCC_DELTA is required}"
: "${MIN_LABELED_SAMPLES:?MIN_LABELED_SAMPLES is required}"
: "${OUTPUT_JSON:?OUTPUT_JSON is required}"
: "${OUTPUT_MD:?OUTPUT_MD is required}"

cd "${MINT_AGENT_ROOT}"
source scripts/sapelo2/load_modules.sh
unset EBPYTHONPREFIXES
export PYTHONPATH=${MINT_AGENT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}

args=(
  --baseline-representation "${BASELINE_REPRESENTATION}"
  --candidate-representation "${CANDIDATE_REPRESENTATION}"
  --min-pcc-delta "${MIN_PCC_DELTA}"
  --min-labeled-samples "${MIN_LABELED_SAMPLES}"
  --output-json "${OUTPUT_JSON}"
  --output-md "${OUTPUT_MD}"
)
IFS=',' read -r -a baseline_paths <<< "${BASELINE_OOF}"
IFS=',' read -r -a candidate_paths <<< "${CANDIDATE_OOF}"
for path in "${baseline_paths[@]}"; do
  args+=(--baseline-oof "${path}")
done
for path in "${candidate_paths[@]}"; do
  args+=(--candidate-oof "${path}")
done

"${PYTHON_BIN}" -m mint_scout.filtration_comparison "${args[@]}"

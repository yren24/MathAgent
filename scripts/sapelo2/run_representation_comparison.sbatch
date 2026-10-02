#!/bin/bash
#SBATCH --job-name=mint-agent-pl-compare
#SBATCH --partition=batch
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2gb
#SBATCH --time=00:10:00
#SBATCH --output=/path/to/workdir/mathagent/logs/%x-%j.out
#SBATCH --error=/path/to/workdir/mathagent/logs/%x-%j.err

set -euo pipefail

export MINT_AGENT_ROOT=${MINT_AGENT_ROOT:-/path/to/MathAgent}
export PYTHON_BIN=${PYTHON_BIN:-python}
export PYTHONPATH=${MINT_AGENT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}

RUN_ROOT=${RUN_ROOT:-/path/to/workdir/mathagent/runs}
REPRESENTATION_SPEC=${REPRESENTATION_SPEC:-${RUN_ROOT}/representation_train_offset-0_limit-all_47823391.json}
ADAPTIVE_CV=${ADAPTIVE_CV:-${RUN_ROOT}/phase4_gbt_train_47996822.json}
SCOUT_EXECUTION=${SCOUT_EXECUTION:-${RUN_ROOT}/combined_scout_execution_47983663.json}
LEGACY_CV=${LEGACY_CV:-${RUN_ROOT}/legacy_pl_train-cv_48005010.json}
LEGACY_FIXED_TEST=${LEGACY_FIXED_TEST:-${RUN_ROOT}/legacy_pl_historical-fixed-test_48005006.json}
FILTRATION_AUDIT=${FILTRATION_AUDIT:-${RUN_ROOT}/filtration_audit_48004997.json}
MIN_DELTA=${MIN_DELTA:-0.005}
OUTPUT_JSON=${OUTPUT_JSON:-${RUN_ROOT}/representation_comparison_pl_adaptive-vs-legacy_v1.json}
OUTPUT_MD=${OUTPUT_MD:-${RUN_ROOT}/representation_comparison_pl_adaptive-vs-legacy_v1.md}

mkdir -p /path/to/workdir/mathagent/logs "${RUN_ROOT}"
cd "${MINT_AGENT_ROOT}"
source scripts/sapelo2/load_modules.sh

"${PYTHON_BIN}" -m mint_scout.representation_comparison \
  --representation-spec "${REPRESENTATION_SPEC}" \
  --adaptive-cv "${ADAPTIVE_CV}" \
  --scout-execution "${SCOUT_EXECUTION}" \
  --legacy-cv "${LEGACY_CV}" \
  --legacy-fixed-test "${LEGACY_FIXED_TEST}" \
  --filtration-audit "${FILTRATION_AUDIT}" \
  --min-delta "${MIN_DELTA}" \
  --output-json "${OUTPUT_JSON}" \
  --output-md "${OUTPUT_MD}"

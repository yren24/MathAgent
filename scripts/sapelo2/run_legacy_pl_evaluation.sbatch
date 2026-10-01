#!/bin/bash
#SBATCH --job-name=mint-agent-legacy-pl-eval
#SBATCH --partition=batch
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64gb
#SBATCH --time=2-00:00:00
#SBATCH --output=/path/to/workdir/mathagent/logs/%x-%j.out
#SBATCH --error=/path/to/workdir/mathagent/logs/%x-%j.err

set -euo pipefail

export MINT_AGENT_ROOT=${MINT_AGENT_ROOT:-/path/to/MathAgent}
export PYTHON_BIN=${PYTHON_BIN:-python}
export PYTHONPATH=${MINT_AGENT_ROOT}/src

MODE=${MODE:-train-cv}
FEATURE_FOLDER=${FEATURE_FOLDER:-/path/to/workdir/mathagent/cache/plbind_features/PL}
SCOUT_EXECUTION=${SCOUT_EXECUTION:-}
RUN_ROOT=${RUN_ROOT:-/path/to/workdir/mathagent/runs}
OUTPUT=${OUTPUT:-${RUN_ROOT}/legacy_pl_${MODE}_${SLURM_JOB_ID}.json}

mkdir -p /path/to/workdir/mathagent/logs "${RUN_ROOT}"
cd "${MINT_AGENT_ROOT}"
source scripts/sapelo2/load_modules.sh

args=(
  --task-config configs/tasks/casf2016_sapelo2.yaml
  --feature-folder "${FEATURE_FOLDER}"
  --mode "${MODE}"
  --output "${OUTPUT}"
)
if [[ "${MODE}" == "train-cv" ]]; then
  args+=(--scout-execution "${SCOUT_EXECUTION}")
fi

"${PYTHON_BIN}" -m mint_scout.evaluate_legacy_pl "${args[@]}"

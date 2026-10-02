#!/bin/bash
#SBATCH --job-name=mint-agent-phase4
#SBATCH --partition=batch
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=24gb
#SBATCH --time=12:00:00
#SBATCH --output=/path/to/workdir/mathagent/logs/%x-%j.out
#SBATCH --error=/path/to/workdir/mathagent/logs/%x-%j.err

set -euo pipefail

export MINT_AGENT_ROOT=${MINT_AGENT_ROOT:-/path/to/MathAgent}
export PYTHON_BIN=${PYTHON_BIN:-python}
export PYTHONPATH=${MINT_AGENT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}

SPLIT=${SPLIT:-train}
OFFSET=${OFFSET:-0}
LIMIT=${LIMIT:-16}
PRIORITY_ORDER=${PRIORITY_ORDER:-PL}
TARGET_VALUE=${TARGET_VALUE:-0.85}
CV_FOLDS=${CV_FOLDS:-5}
RUN_ROOT=${RUN_ROOT:-/path/to/workdir/mathagent/runs}

mkdir -p /path/to/workdir/mathagent/logs /path/to/workdir/mathagent/cache/plbind_features "${RUN_ROOT}"
cd "${MINT_AGENT_ROOT}"
source scripts/sapelo2/load_modules.sh

REPORT=${REPORT:-${RUN_ROOT}/phase4_casf_${SPLIT}_offset-${OFFSET}_limit-${LIMIT}_${SLURM_JOB_ID}.json}

"${PYTHON_BIN}" -m mint_scout.execute_casf \
  --config configs/tasks/casf2016_sapelo2.yaml \
  --gbt-config configs/gbt/plbind_fixed_gbt.yaml \
  --split "${SPLIT}" \
  --offset "${OFFSET}" \
  --limit "${LIMIT}" \
  --priority-order "${PRIORITY_ORDER}" \
  --target-value "${TARGET_VALUE}" \
  --cv-folds "${CV_FOLDS}" \
  --output "${REPORT}"

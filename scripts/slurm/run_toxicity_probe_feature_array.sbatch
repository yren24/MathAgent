#!/bin/bash
#SBATCH --job-name=mint-tox-feature
#SBATCH --partition=batch
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=08:00:00
#SBATCH --output=/path/to/workdir/mathagent/logs/toxicity-feature-%A_%a.out
#SBATCH --error=/path/to/workdir/mathagent/logs/toxicity-feature-%A_%a.err

set -euo pipefail

MINT_AGENT_ROOT=${MINT_AGENT_ROOT:-/path/to/MathAgent}
BASE_DESIGN_ROOT=${BASE_DESIGN_ROOT:-/path/to/workdir/mathagent/toxicity/design-v1}
PROBE_ROOT=${PROBE_ROOT:-/path/to/workdir/mathagent/toxicity/probe-v1}
PLAN_ROOT=${PLAN_ROOT:-${PROBE_ROOT}/features}
FEATURE_ROOT=${FEATURE_ROOT:-/path/to/workdir/mathagent/toxicity/feature-cache}
LEGACY_ROOT=${LEGACY_ROOT:-/path/to/workdir/mathagent/external/toxicity-phase0/legacy}
DATASET_ROOT=${DATASET_ROOT:-/path/to/workdir/mathagent/external/toxicity-phase0/LD50}
DATASET_ID=${DATASET_ID:-LD50}
MOLECULE_DIRS=${MOLECULE_DIRS:-${DATASET_ROOT}/LD50_train_x}
AGENT_ENV=${AGENT_ENV:-/path/to/MathAgent/.venv-langgraph}
PYTHON_BIN=${PYTHON_BIN:-${AGENT_ENV}/bin/python}
GUDHI_ROOT=${GUDHI_ROOT:-/path/to/workdir/mathagent/deps/gudhi-3.13.0}

mkdir -p /path/to/workdir/mathagent/logs "${FEATURE_ROOT}"
cd "${MINT_AGENT_ROOT}"
source scripts/sapelo2/load_modules.sh
unset EBPYTHONPREFIXES
export PYTHONPATH="${MINT_AGENT_ROOT}/src:${GUDHI_ROOT}"

IFS=: read -r -a molecule_dirs <<< "${MOLECULE_DIRS}"
molecule_args=()
for molecule_dir in "${molecule_dirs[@]}"; do
  [[ -n "${molecule_dir}" ]] && molecule_args+=(--molecule-dir "${molecule_dir}")
done

"${PYTHON_BIN}" -m mint_scout.toxicity.feature_batch \
  --plan "${PLAN_ROOT}/toxicity_feature_plan.json" \
  --task-index "${SLURM_ARRAY_TASK_ID}" \
  --probe-selection "${PROBE_ROOT}/toxicity_probe_selection.json" \
  --dataset-manifest "${BASE_DESIGN_ROOT}/ld50_manifest.csv" \
  --legacy-root "${LEGACY_ROOT}" \
  "${molecule_args[@]}" \
  --feature-root "${FEATURE_ROOT}" \
  --dataset-id "${DATASET_ID}"

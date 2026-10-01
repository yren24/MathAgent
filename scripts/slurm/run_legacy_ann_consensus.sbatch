#!/bin/bash
#SBATCH --job-name=mint-legacy-ann-consensus
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

set -euo pipefail

LEGACY_ROOT=${LEGACY_ROOT:?LEGACY_ROOT must be provided}
SELECTED_CSV=${SELECTED_CSV:?SELECTED_CSV must be provided}
OUTPUT_DIR=${OUTPUT_DIR:?OUTPUT_DIR must be provided}

module purge
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1

python "${LEGACY_ROOT}/snn/generate_ensemble_results.py" \
  --root "${LEGACY_ROOT}" \
  --selected-csv "${SELECTED_CSV}" \
  --out "${OUTPUT_DIR}" \
  --year 2016 \
  --top-n 10000

#!/bin/bash
#SBATCH --job-name=mint-legacy-ann
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6

set -euo pipefail

MINT_AGENT_ROOT=${MINT_AGENT_ROOT:?MINT_AGENT_ROOT must be provided}
LEGACY_ROOT=${LEGACY_ROOT:?LEGACY_ROOT must be provided}
INDEX_ROOT=${INDEX_ROOT:?INDEX_ROOT must be provided}
INVARIANT=${INVARIANT:?INVARIANT must be provided}
TRAIN_MANIFEST=${TRAIN_MANIFEST:?TRAIN_MANIFEST must be provided}
TEST_MANIFEST=${TEST_MANIFEST:?TEST_MANIFEST must be provided}
OUTPUT_DIR=${OUTPUT_DIR:?OUTPUT_DIR must be provided}
LEARNING_RATE=${LEARNING_RATE:-8e-5}

module purge
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1 scikit-learn/1.3.1-gfbf-2023a

cd "${MINT_AGENT_ROOT}"
python scripts/diagnostics/run_legacy_ann_compat.py \
  --legacy-root "${LEGACY_ROOT}" \
  --index-root "${INDEX_ROOT}" \
  --invariant "${INVARIANT}" \
  --train-manifest "${TRAIN_MANIFEST}" \
  --test-manifest "${TEST_MANIFEST}" \
  --output-dir "${OUTPUT_DIR}" \
  --learning-rate "${LEARNING_RATE}"

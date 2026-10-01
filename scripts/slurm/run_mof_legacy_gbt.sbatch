#!/bin/bash
#SBATCH --job-name=mint-mof-gbt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32gb
#SBATCH --time=04:00:00

set -euo pipefail

: "${MINT_AGENT_ROOT:?Set MINT_AGENT_ROOT}"
: "${MOF_LEGACY_ROOT:?Set MOF_LEGACY_ROOT}"
: "${MOF_TOPOLOGY:?Set MOF_TOPOLOGY}"
: "${MOF_PROPERTY:?Set MOF_PROPERTY}"
: "${MOF_DATA_DIR:?Set MOF_DATA_DIR}"
: "${MOF_FEATURE_DIR:?Set MOF_FEATURE_DIR}"
: "${MOF_RESULT_DIR:?Set MOF_RESULT_DIR}"

source "${MINT_AGENT_ROOT}/scripts/sapelo2/load_modules.sh"
unset EBPYTHONPREFIXES
export PYTHONPATH="${MINT_AGENT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

python "${MOF_LEGACY_ROOT}/mof_topology_gbt.py" \
  --topology "${MOF_TOPOLOGY}" \
  --property "${MOF_PROPERTY}" \
  --data-dir "${MOF_DATA_DIR}" \
  --feature-dir "${MOF_FEATURE_DIR}" \
  --output-dir "${MOF_RESULT_DIR}" \
  --folds "${MOF_FOLDS:-5}" \
  --repeat "${MOF_REPEAT:-0}"

stem="${MOF_PROPERTY}_${MOF_TOPOLOGY}_repeat${MOF_REPEAT:-0}_gbt"
python -m mint_scout.mof.standard_metrics \
  --predictions "${MOF_RESULT_DIR}/${stem}_predictions.csv"

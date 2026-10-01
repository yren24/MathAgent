#!/bin/bash
#SBATCH --job-name=mint-mof-feature
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16gb
#SBATCH --time=08:00:00

set -euo pipefail

: "${MINT_AGENT_ROOT:?Set MINT_AGENT_ROOT}"
: "${MOF_LEGACY_ROOT:?Set MOF_LEGACY_ROOT}"
: "${MOF_TOPOLOGY:?Set MOF_TOPOLOGY}"
: "${MOF_PROPERTY:?Set MOF_PROPERTY}"
: "${MOF_CIF_DIR:?Set MOF_CIF_DIR}"
: "${MOF_DATA_DIR:?Set MOF_DATA_DIR}"
: "${MOF_FEATURE_DIR:?Set MOF_FEATURE_DIR}"

source "${MINT_AGENT_ROOT}/scripts/sapelo2/load_modules.sh"
module load pymatgen/2023.12.18-foss-2023a
unset EBPYTHONPREFIXES
MOF_PYTHON_BIN=${MOF_PYTHON_BIN:-python}

args=(
  "${MOF_PYTHON_BIN}" "${MOF_LEGACY_ROOT}/mof_topology_features_forman120_final.py"
  --topology "${MOF_TOPOLOGY}"
  --property "${MOF_PROPERTY}"
  --all
  --start "${MOF_START:-0}"
  --cif-dir "${MOF_CIF_DIR}"
  --data-dir "${MOF_DATA_DIR}"
  --output-dir "${MOF_FEATURE_DIR}"
  --xyz-dir "${MOF_XYZ_DIR:-${MOF_FEATURE_DIR}/xyz}"
)
if [[ -n "${MOF_END:-}" ]]; then
  args+=(--end "${MOF_END}")
fi
"${args[@]}"

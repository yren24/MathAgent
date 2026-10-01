#!/bin/bash
#SBATCH --job-name=mint-mof-category-audit
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16gb
#SBATCH --time=04:00:00

set -euo pipefail

: "${MINT_AGENT_ROOT:?Set MINT_AGENT_ROOT}"
: "${MOF_MANIFEST:?Set MOF_MANIFEST}"
: "${MOF_CATEGORY_AUDIT_OUTPUT:?Set MOF_CATEGORY_AUDIT_OUTPUT}"

source "${MINT_AGENT_ROOT}/scripts/sapelo2/load_modules.sh"
module load pymatgen/2023.12.18-foss-2023a
unset EBPYTHONPREFIXES
export PYTHONPATH="${MINT_AGENT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

args=(
  python -m mint_scout.mof.category_audit
  --manifest "${MOF_MANIFEST}"
  --output "${MOF_CATEGORY_AUDIT_OUTPUT}"
)
if [[ -n "${MOF_CATEGORY_SCHEMA:-}" ]]; then
  args+=(--schema "${MOF_CATEGORY_SCHEMA}")
fi
"${args[@]}"

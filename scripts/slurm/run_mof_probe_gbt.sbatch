#!/bin/bash
#SBATCH --job-name=mint-mof-probe-gbt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32gb
#SBATCH --time=04:00:00

set -euo pipefail

: "${MINT_AGENT_ROOT:?Set MINT_AGENT_ROOT}"
: "${MOF_MANIFEST:?Set MOF_MANIFEST}"
: "${MOF_PROBE_IDS:?Set MOF_PROBE_IDS}"
: "${MOF_FEATURE_DIR:?Set MOF_FEATURE_DIR}"
: "${MOF_TOPOLOGY:?Set MOF_TOPOLOGY}"
: "${MOF_PROPERTY:?Set MOF_PROPERTY}"
: "${MOF_RESULT_DIR:?Set MOF_RESULT_DIR}"

source "${MINT_AGENT_ROOT}/scripts/sapelo2/load_modules.sh"
unset EBPYTHONPREFIXES
export PYTHONPATH="${MINT_AGENT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

if [[ "${MOF_REQUIRE_FULL_FEATURES:-0}" == "1" ]]; then
  : "${MOF_EXPECTED_FEATURE_COUNT:?Set MOF_EXPECTED_FEATURE_COUNT when requiring full features}"
  feature_path="${MOF_FEATURE_DIR}/${MOF_TOPOLOGY}/${MOF_PROPERTY}"
  actual_count=$(find "${feature_path}" -maxdepth 1 -type f -name "*.npy" | wc -l)
  if [[ "${actual_count}" -ne "${MOF_EXPECTED_FEATURE_COUNT}" ]]; then
    echo "Expected ${MOF_EXPECTED_FEATURE_COUNT} feature files, found ${actual_count} in ${feature_path}" >&2
    exit 2
  fi
fi

python -m mint_scout.mof.probe_gbt \
  --manifest "${MOF_MANIFEST}" \
  --probe-ids "${MOF_PROBE_IDS}" \
  --feature-dir "${MOF_FEATURE_DIR}" \
  --topology "${MOF_TOPOLOGY}" \
  --property "${MOF_PROPERTY}" \
  --output-dir "${MOF_RESULT_DIR}" \
  --folds "${MOF_FOLDS:-5}" \
  --repeats "${MOF_REPEATS:-3}" \
  --seed "${MOF_SEED:-2026}"

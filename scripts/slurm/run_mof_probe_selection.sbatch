#!/bin/bash
#SBATCH --job-name=mint-mof-probe-selection
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4gb
#SBATCH --time=00:30:00

set -euo pipefail

: "${MINT_AGENT_ROOT:?Set MINT_AGENT_ROOT}"
: "${MOF_MANIFEST:?Set MOF_MANIFEST}"
: "${MOF_FEATURE_DIR:?Set MOF_FEATURE_DIR}"
: "${MOF_LEGACY_RESULT_DIR:?Set MOF_LEGACY_RESULT_DIR}"
: "${MOF_PROPERTY:?Set MOF_PROPERTY}"
: "${MOF_WORKFLOW_OUTPUT:?Set MOF_WORKFLOW_OUTPUT}"
: "${MOF_PROBE_RESULT_DIR:?Set MOF_PROBE_RESULT_DIR}"
: "${MOF_SELECTION_OUTPUT:?Set MOF_SELECTION_OUTPUT}"

source "${MINT_AGENT_ROOT}/scripts/sapelo2/load_modules.sh"
unset EBPYTHONPREFIXES
export PYTHONPATH="${MINT_AGENT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

python -m mint_scout.mof.agent_workflow \
  --manifest "${MOF_MANIFEST}" \
  --feature-dir "${MOF_FEATURE_DIR}" \
  --result-dir "${MOF_LEGACY_RESULT_DIR}" \
  --property "${MOF_PROPERTY}" \
  --probe-size "${MOF_PROBE_SIZE:-750}" \
  --probe-strata "${MOF_PROBE_STRATA:-10}" \
  --seed "${MOF_SEED:-2026}" \
  --output "${MOF_WORKFLOW_OUTPUT}"

python -m mint_scout.mof.selection \
  --workflow "${MOF_WORKFLOW_OUTPUT}" \
  --probe-result-dir "${MOF_PROBE_RESULT_DIR}" \
  --output "${MOF_SELECTION_OUTPUT}" \
  --r2-tolerance "${MOF_R2_TOLERANCE:-0.01}" \
  --stability-tolerance "${MOF_STABILITY_TOLERANCE:-0.005}"

if [[ "${MOF_REQUIRE_FINAL_SELECTION:-0}" == "1" ]]; then
  python - "${MOF_SELECTION_OUTPUT}" <<'PY'
import json
import sys

payload = json.load(open(sys.argv[1], encoding="utf-8"))
if not payload.get("is_final_selection"):
    raise SystemExit("MOF final selection is not ready: one or more probe reports are missing or invalid")
PY
fi

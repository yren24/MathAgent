#!/bin/bash
set -euo pipefail

MINT_AGENT_ROOT=${MINT_AGENT_ROOT:-/path/to/MathAgent}
SUMMARY_REPORT=${SUMMARY_REPORT:-/path/to/workdir/mathagent/runs/pipelines/atom3d-lba-onboarding-live-tolerated_atom3d-lba-onboarding-live-tolerated.artifacts/pipeline_summary.json}
EXPLANATION_LANGUAGE=${EXPLANATION_LANGUAGE:-English}

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "OPENAI_API_KEY must be exported in this shell before submitting." >&2
  exit 2
fi
if [[ -z "${OPENAI_MODEL:-}" ]]; then
  echo "OPENAI_MODEL must be exported in this shell before submitting." >&2
  exit 2
fi
if [[ ! -f "${SUMMARY_REPORT}" ]]; then
  echo "SUMMARY_REPORT does not exist: ${SUMMARY_REPORT}" >&2
  exit 2
fi

cd "${MINT_AGENT_ROOT}"
export SUMMARY_REPORT
export EXPLANATION_LANGUAGE
sbatch scripts/sapelo2/run_llm_explanation.sbatch

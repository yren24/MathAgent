#!/bin/bash
# Deterministic preparation of the MOF agent workflow.  This does not rerun
# legacy feature generation; it discovers and reuses cached feature artifacts.

set -euo pipefail

ROOT=/path/to/MathAgent
SCRATCH=/path/to/workdir/mathagent/mof
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

python -m mint_scout.mof.agent_workflow \
  --manifest "$SCRATCH/manifests/O2uptakemolkg.csv" \
  --feature-dir "$SCRATCH/features/o2_legacy_v1" \
  --result-dir "$SCRATCH/runs/o2_uptake_legacy_parity_v1" \
  --property O2uptakemolkg \
  --probe-size 750 \
  --probe-strata 10 \
  --seed 2026 \
  --output "$SCRATCH/runs/o2_uptake_agent_v1/workflow_preflight.json"

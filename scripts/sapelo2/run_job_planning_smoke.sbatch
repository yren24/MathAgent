#!/bin/bash
#SBATCH --job-name=mint-agent-job-plan
#SBATCH --partition=batch
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2gb
#SBATCH --time=00:10:00
#SBATCH --output=/path/to/workdir/mathagent/logs/%x-%j.out
#SBATCH --error=/path/to/workdir/mathagent/logs/%x-%j.err

set -euo pipefail

MINT_AGENT_ROOT=${MINT_AGENT_ROOT:-/path/to/MathAgent}
PYTHON_BIN=${PYTHON_BIN:-python}
REGISTRY=${REGISTRY:-/path/to/workdir/mathagent/memory/artifacts.sqlite}
RUN_ROOT=${RUN_ROOT:-/path/to/workdir/mathagent/runs}
TASK_CONFIG=${TASK_CONFIG:-configs/tasks/casf2016_sapelo2.yaml}
EXECUTION_PROFILE=${EXECUTION_PROFILE:-configs/execution/sapelo2.yaml}
REPRESENTATION_SPEC=${REPRESENTATION_SPEC:-/path/to/workdir/mathagent/runs/representation_train_offset-0_limit-all_47823391.json}
DATASET_ID=${DATASET_ID:-casf2016}
INVARIANT=${INVARIANT:-FPRC}
THREAD_ID=${THREAD_ID:-job-planning-smoke-${SLURM_JOB_ID}}
OUTPUT=${OUTPUT:-${RUN_ROOT}/agent_job_plan_${SLURM_JOB_ID}.json}
LEDGER=${LEDGER:-${RUN_ROOT}/agent_job_ledger_${SLURM_JOB_ID}.json}

mkdir -p /path/to/workdir/mathagent/logs "${RUN_ROOT}"
cd "${MINT_AGENT_ROOT}"
source scripts/sapelo2/load_modules.sh
unset EBPYTHONPREFIXES
export PYTHONPATH=${MINT_AGENT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}

"${PYTHON_BIN}" -m mint_scout.run_agent_graph \
  --task-config "${TASK_CONFIG}" \
  --registry "${REGISTRY}" \
  --dataset-id "${DATASET_ID}" \
  --invariant "${INVARIANT}" \
  --evidence-scope full_train \
  --execution-profile "${EXECUTION_PROFILE}" \
  --representation-spec "${REPRESENTATION_SPEC}" \
  --thread-id "${THREAD_ID}" \
  --output "${OUTPUT}"

"${PYTHON_BIN}" -m mint_scout.manage_jobs submit \
  --plan "${OUTPUT}" \
  --ledger "${LEDGER}"

"${PYTHON_BIN}" - "${OUTPUT}" "${LEDGER}" <<'PY'
import json
import sys

plan = json.load(open(sys.argv[1], encoding="utf-8"))
ledger = json.load(open(sys.argv[2], encoding="utf-8"))
assert plan["status"] == "NEEDS_FEATURE_COMPUTE"
assert len(plan["feature_jobs"]) == 1
assert plan["feature_jobs"][0]["environment"]["INVARIANT"] == "FPRC"
assert ledger["status"] == "PLANNED"
assert ledger["jobs"][0]["attempts"] == []
print("dry_run_verified=1 plan=" + sys.argv[1] + " ledger=" + sys.argv[2])
PY

#!/bin/bash
#SBATCH --job-name=mint-agent-graph
#SBATCH --partition=batch
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2gb
#SBATCH --time=00:10:00
#SBATCH --output=/path/to/workdir/mathagent/logs/%x-%j.out
#SBATCH --error=/path/to/workdir/mathagent/logs/%x-%j.err

set -euo pipefail

export MINT_AGENT_ROOT=${MINT_AGENT_ROOT:-/path/to/MathAgent}
export PYTHON_BIN=${PYTHON_BIN:-${MINT_AGENT_ROOT}/.venv-langgraph/bin/python}

REGISTRY=${REGISTRY:-/path/to/workdir/mathagent/memory/artifacts.sqlite}
RUN_ROOT=${RUN_ROOT:-/path/to/workdir/mathagent/runs}
TASK_CONFIG=${TASK_CONFIG:-configs/tasks/casf2016_sapelo2.yaml}
EXECUTION_PROFILE=${EXECUTION_PROFILE:-configs/execution/sapelo2.yaml}
DATASET_ID=${DATASET_ID:-casf2016}
INVARIANTS=${INVARIANTS:-PL}
EVIDENCE_SCOPE=${EVIDENCE_SCOPE:-full_train}
REPRESENTATION_SPEC=${REPRESENTATION_SPEC:-}
SAMPLE_ID_FILE=${SAMPLE_ID_FILE:-}
THREAD_ID=${THREAD_ID:-casf2016-pl-cache-replay-v1}
OUTPUT=${OUTPUT:-${RUN_ROOT}/agent_graph_casf2016_pl_cache-replay_v1.json}
OUTPUT_MERMAID=${OUTPUT_MERMAID:-${RUN_ROOT}/agent_graph_casf2016_pl_cache-replay_v1.mmd}

mkdir -p /path/to/workdir/mathagent/logs "${RUN_ROOT}"
cd "${MINT_AGENT_ROOT}"
source scripts/sapelo2/load_modules.sh
unset EBPYTHONPREFIXES
export PYTHONPATH=${MINT_AGENT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}

args=(
  --task-config "${TASK_CONFIG}"
  --registry "${REGISTRY}"
  --dataset-id "${DATASET_ID}"
  --evidence-scope "${EVIDENCE_SCOPE}"
  --execution-profile "${EXECUTION_PROFILE}"
  --thread-id "${THREAD_ID}"
  --output "${OUTPUT}"
  --output-mermaid "${OUTPUT_MERMAID}"
)
for invariant in ${INVARIANTS}; do
  args+=(--invariant "${invariant}")
done
if [[ -n "${REPRESENTATION_SPEC}" ]]; then
  args+=(--representation-spec "${REPRESENTATION_SPEC}")
fi
if [[ -n "${SAMPLE_ID_FILE}" ]]; then
  args+=(--sample-id-file "${SAMPLE_ID_FILE}")
fi

"${PYTHON_BIN}" -m mint_scout.run_agent_graph "${args[@]}"

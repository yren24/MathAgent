#!/bin/bash
#SBATCH --job-name=mint-agent-tests
#SBATCH --partition=batch
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8gb
#SBATCH --time=00:30:00
#SBATCH --output=/path/to/workdir/mathagent/logs/%x-%j.out
#SBATCH --error=/path/to/workdir/mathagent/logs/%x-%j.err

set -euo pipefail

MINT_AGENT_ROOT=${MINT_AGENT_ROOT:-/path/to/MathAgent}
AGENT_ENV=${AGENT_ENV:-${MINT_AGENT_ROOT}/.venv-langgraph}
RUN_ROOT=${RUN_ROOT:-/path/to/workdir/mathagent/runs}

mkdir -p /path/to/workdir/mathagent/logs "${RUN_ROOT}"
cd "${MINT_AGENT_ROOT}"
source scripts/sapelo2/load_modules.sh
unset EBPYTHONPREFIXES
export PYTHONPATH=${MINT_AGENT_ROOT}/src
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1

"${AGENT_ENV}/bin/python" -m compileall -q src/mint_scout
"${AGENT_ENV}/bin/python" -m py_compile \
  scripts/sapelo2/make_manifest_split_subset.py \
  scripts/sapelo2/make_casf_manifest.py
bash -n \
  scripts/slurm/run_dataset_audit.sbatch \
  scripts/slurm/run_representation_design.sbatch \
  scripts/slurm/run_probe_selection.sbatch \
  scripts/slurm/run_probe_audit.sbatch \
  scripts/slurm/run_feature_batch.sbatch \
  scripts/slurm/run_feature_qc.sbatch \
  scripts/slurm/run_filtration_audit.sbatch \
  scripts/slurm/run_scout_oof.sbatch \
  scripts/slurm/run_scout_combine.sbatch \
  scripts/slurm/run_model_evaluation.sbatch \
  scripts/slurm/run_agent_lifecycle.sbatch \
  scripts/sapelo2/run_feature_batch.sbatch \
  scripts/sapelo2/prepare_bdb2020plus.sbatch \
  scripts/sapelo2/run_bdb2020plus_external_gbt.sbatch \
  scripts/sapelo2/run_agent_graph.sbatch \
  scripts/sapelo2/run_agent_lifecycle.sbatch \
  scripts/sapelo2/run_llm_explanation.sbatch \
  scripts/sapelo2/submit_llm_explanation.sh \
  scripts/sapelo2/run_filtration_comparison.sbatch \
  scripts/sapelo2/register_artifact.sbatch \
  scripts/sapelo2/preview_hiqbind_subset.sbatch
"${AGENT_ENV}/bin/python" -m pytest -q \
  --junitxml="${RUN_ROOT}/pytest_cpu_${SLURM_JOB_ID}.xml"
